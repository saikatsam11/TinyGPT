"""
finetune.py — Instruction fine-tuning for the custom GPT model

"""

import os, math, json, time, sys
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer

from config import ModelConfig
from gpt import GPT


# ── Fine-tune hyperparameters ────────────────────────────────────────────────
PRETRAINED_CKPT = "ckpt_best.pt"
DATA_PATH       = "final_data_cleaned.jsonl"
TOKENIZER_PATH  = "tokenizer/tokenizer.json"
CKPT_DIR        = "checkpoints_ft"

BATCH_SIZE   = 16
GRAD_ACCUM   = 2        # effective batch = 32 sequences
LR           = 6e-5    # lower lr for more careful fine-tuning
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
WARMUP_STEPS = 100
MAX_STEPS    = 1500
VAL_RATIO    = 0.05
LOG_EVERY    = 20
VAL_EVERY    = 100


# ── Prompt template ──────────────────────────────────────────────────────────
def build_prefix_and_full(instruction: str, inp: str, output: str) -> tuple[str, str]:
    """
    Returns (prefix, full_text).
    prefix  = everything up to and including '### Response:\n'
    full    = prefix + output
    Loss is masked to zero on prefix tokens.
    """
    if inp.strip():
        prefix = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prefix = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prefix, prefix + output


# ── Dataset ──────────────────────────────────────────────────────────────────
class InstructionDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, context_len: int):
        self.context_len = context_len
        bos_id = tokenizer.token_to_id("<|bos|>")
        eos_id = tokenizer.token_to_id("<|eos|>")

        self.samples: list[tuple[list[int], list[int]]] = []
        skipped = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                prefix, full = build_prefix_and_full(
                    item.get("instruction", ""),
                    item.get("input", ""),
                    item.get("output", ""),
                )

                prefix_ids = tokenizer.encode(prefix).ids
                full_ids   = tokenizer.encode(full).ids

                # Full sequence: BOS + full tokens + EOS
                seq = [bos_id] + full_ids + [eos_id]

                # Truncate to fit model context (keep room for the shift: seq[:-1] / seq[1:])
                max_len = context_len + 1
                if len(seq) > max_len:
                    seq = seq[:max_len]

                # n_prefix: number of tokens in [BOS + prefix] — loss is masked here
                n_prefix = 1 + len(prefix_ids)

                # Skip if truncation swallowed all response tokens
                if n_prefix >= len(seq) - 1:
                    skipped += 1
                    continue

                # Build labels: -1 (ignore) for instruction, real token id for response
                # label[i] = what the model should predict at position i → seq[i+1]
                labels = [-1] * len(seq)
                for i in range(n_prefix - 1, len(seq) - 1):
                    labels[i] = seq[i + 1]

                # Input and target are the standard LM shift (drop last)
                self.samples.append((seq[:-1], labels[:-1]))

        print(f"  Dataset: {len(self.samples):,} samples | {skipped} skipped (too long / empty)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch: list, pad_id: int):
    """Pad variable-length sequences to the longest in the batch."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    y_pad = torch.full((len(ys), max_len), -1,     dtype=torch.long)  # -1 = ignore
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x
        y_pad[i, :y.size(0)] = y
    return x_pad, y_pad


# ── LR schedule (cosine with warmup) ────────────────────────────────────────
def get_lr(step: int) -> float:
    min_lr = LR * 0.1
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return min_lr
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (LR - min_lr)


# ── Checkpoint helpers ───────────────────────────────────────────────────────
def save_ckpt(model, optimizer, step: int, val_loss: float, tag: str):
    os.makedirs(CKPT_DIR, exist_ok=True)
    raw  = getattr(model, "_orig_mod", model)
    path = os.path.join(CKPT_DIR, f"ft_ckpt_{tag}.pt")
    torch.save({"step": step, "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(), "val_loss": val_loss}, path)
    print(f"  Saved → {path}  (step={step}  val_loss={val_loss:.4f})")


# ── Validation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-1,
            )
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    pad_id    = tokenizer.token_to_id("<|pad|>")
    print(f"Tokenizer loaded — vocab_size={tokenizer.get_vocab_size():,}")

    # Dataset
    cfg      = ModelConfig()
    full_ds  = InstructionDataset(DATA_PATH, tokenizer, cfg.context_len)
    n_val    = max(1, int(len(full_ds) * VAL_RATIO))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    collate = partial(collate_fn, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=2, pin_memory=True)

    # Model — load pretrained weights
    model = GPT(cfg).to(device)
    if os.path.exists(PRETRAINED_CKPT):
        ckpt  = torch.load(PRETRAINED_CKPT, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded pretrained weights <- {PRETRAINED_CKPT}")
    else:
        print(f"WARNING: {PRETRAINED_CKPT} not found — fine-tuning from random init")

    # Lower dropout for fine-tuning
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.05

    # Optimizer — weight decay on matrices only
    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,    "weight_decay": WEIGHT_DECAY},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )

    if hasattr(torch, "compile") and sys.platform != "win32":
        print("Compiling model with torch.compile() ...")
        model = torch.compile(model)

    print(f"\n{'='*60}")
    print(f"  Instruction Fine-tuning")
    print(f"  Device      : {device}")
    print(f"  Train       : {n_train}  |  Val : {n_val}")
    print(f"  Batch       : {BATCH_SIZE} × grad_accum {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} eff.")
    print(f"  LR          : {LR}  |  Steps : {MAX_STEPS}")
    print(f"  Checkpoints : {CKPT_DIR}/")
    print(f"{'='*60}\n")

    model.train()
    loader_iter = iter(train_loader)
    best_val    = float("inf")
    t_start     = time.perf_counter()

    for step in range(MAX_STEPS):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                # Masked loss: y == -1 positions are skipped (instruction tokens + padding)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                )

            (loss / GRAD_ACCUM).backward()
            loss_accum += loss.item() / GRAD_ACCUM

        raw = getattr(model, "_orig_mod", model)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0:
            elapsed = (time.perf_counter() - t_start) / 60
            print(f"step {step:4d}/{MAX_STEPS} | loss {loss_accum:.4f} | "
                  f"ppl {math.exp(min(loss_accum, 20)):7.2f} | "
                  f"lr {lr:.2e} | elapsed {elapsed:.1f}m")

        if step > 0 and step % VAL_EVERY == 0:
            val_loss = validate(model, val_loader, device)
            print(f"\n  ┌── VALIDATION  step {step} ──────────────────────")
            print(f"  │  val_loss={val_loss:.4f}  ppl={math.exp(min(val_loss, 20)):.2f}")
            save_ckpt(model, optimizer, step, val_loss, "latest")
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(model, optimizer, step, val_loss, "best")
                print(f"  │  ✓ New best checkpoint!")
            print(f"  └────────────────────────────────────────────────\n")

    save_ckpt(model, optimizer, MAX_STEPS, best_val, "final")
    total_min = (time.perf_counter() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete!")
    print(f"  Total time  : {total_min:.1f}m")
    print(f"  Best val    : {best_val:.4f}  (ppl {math.exp(min(best_val, 20)):.2f})")
    print(f"  Checkpoints : {CKPT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
