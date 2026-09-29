"""
finetune.py — Instruction fine-tuning for the custom GPT model
───────────────────────────────────────────────────────────────
Loads ckpt_best.pt and fine-tunes on cleaned_data_final.jsonl
using masked next-token prediction (loss only on response tokens).
"""

# ── Imports ──────────────────────────────────────────────────────────────────

import os           # for file/directory operations (makedirs, path.exists)
import math         # for math.cos, math.exp in LR schedule and PPL
import json         # for parsing JSONL dataset lines
import time         # for measuring training duration
import sys          # for sys.platform check (torch.compile on Windows)
from functools import partial  # for creating collate_fn with fixed pad_id argument

import torch                          # main PyTorch library
import torch.nn.functional as F       # for F.cross_entropy loss function
from torch.utils.data import (
    Dataset,      # base class for custom datasets
    DataLoader,   # batches and loads data efficiently
    random_split  # splits dataset into train/val randomly
)
from tokenizers import Tokenizer      # HuggingFace fast tokenizer

from config import ModelConfig        # our ModelConfig dataclass (all hyperparameters)
from gpt    import GPT                # our GPT model class


# ── Fine-tune hyperparameters ────────────────────────────────────────────────

PRETRAINED_CKPT = "ckpt_best.pt"            # path to pretrained model checkpoint
DATA_PATH       = "final_data_cleaned.jsonl" # path to instruction dataset
TOKENIZER_PATH  = "tokenizer/tokenizer.json" # path to trained BPE tokenizer
CKPT_DIR        = "checkpoints_ft"           # directory to save fine-tuned checkpoints

BATCH_SIZE   = 16     # sequences per forward pass
GRAD_ACCUM   = 2      # gradient accumulation steps → effective batch = 16×2 = 32
LR           = 6e-5   # peak learning rate (10× lower than pretraining 6e-4)
WEIGHT_DECAY = 0.1    # L2 regularisation on weight matrices only
GRAD_CLIP    = 1.0    # maximum gradient norm to prevent exploding gradients
WARMUP_STEPS = 100    # steps to linearly increase LR from 0 to peak
MAX_STEPS    = 1500   # total fine-tuning steps (~8.2 epochs over 2922 train samples)
VAL_RATIO    = 0.05   # fraction of dataset used for validation (5% = 153 samples)
LOG_EVERY    = 20     # print training log every 20 steps
VAL_EVERY    = 100    # run validation every 100 steps


# ── Prompt template ──────────────────────────────────────────────────────────

def build_prefix_and_full(instruction: str, inp: str, output: str) -> tuple[str, str]:
    """
    Builds two strings:
      prefix    = instruction part only (up to ### Response:)
      full_text = prefix + output (complete sequence)

    Loss is only computed on output tokens (response part)
    prefix tokens are masked with -1 during training
    """

    if inp.strip():
        # Sample HAS an input field (229 out of 3075 samples)
        # Format: Instruction + Input + Response header
        prefix = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        # Sample has NO input field (2846 out of 3075 samples)
        # Format: Instruction + Response header only
        prefix = f"### Instruction:\n{instruction}\n\n### Response:\n"

    # full = prefix + actual output text
    # This is what gets tokenised and fed to the model
    return prefix, prefix + output


# ── Dataset ──────────────────────────────────────────────────────────────────

class InstructionDataset(Dataset):
    """
    Custom PyTorch Dataset for instruction tuning.

    Key feature: Response-only loss masking
      - Instruction tokens → label = -1 (ignored in cross_entropy)
      - Response tokens    → label = actual next token ID (contributes to loss)

    This teaches the model to generate responses
    without memorising instruction formats
    """

    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, context_len: int):
        self.context_len = context_len  # maximum sequence length (512 for fine-tuning)

        # Get special token IDs from tokenizer
        bos_id = tokenizer.token_to_id("<|bos|>")  # beginning of sequence token ID
        eos_id = tokenizer.token_to_id("<|eos|>")  # end of sequence token ID

        self.samples: list[tuple[list[int], list[int]]] = []  # stores (input_ids, labels) pairs
        skipped = 0  # counter for skipped samples (too long or empty)

        # Read JSONL file line by line
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()    # remove whitespace and newlines
                if not line:           # skip empty lines
                    continue

                item = json.loads(line)  # parse JSON string → Python dict

                # Build prefix (instruction part) and full text (instruction + output)
                prefix, full = build_prefix_and_full(
                    item.get("instruction", ""),  # instruction text
                    item.get("input", ""),         # optional input context
                    item.get("output", ""),        # expected response
                )

                # Tokenise prefix only → to know where response starts
                prefix_ids = tokenizer.encode(prefix).ids  # list of token IDs for prefix

                # Tokenise full text (instruction + response)
                full_ids = tokenizer.encode(full).ids  # list of token IDs for full text

                # Build complete sequence: BOS + full tokens + EOS
                seq = [bos_id] + full_ids + [eos_id]
                # Example: [1, 234, 567, 890, ..., 2]
                #           BOS  ← full text tokens →  EOS

                # Truncate if sequence is too long for model context window
                max_len = context_len + 1  # +1 because we need to shift for targets
                if len(seq) > max_len:
                    seq = seq[:max_len]  # keep first max_len tokens only

                # n_prefix = number of tokens in [BOS + prefix]
                # these tokens will be MASKED from loss computation
                n_prefix = 1 + len(prefix_ids)  # 1 for BOS token

                # Skip sample if truncation removed all response tokens
                # (nothing left to learn from)
                if n_prefix >= len(seq) - 1:
                    skipped += 1
                    continue

                # ── Build loss mask (labels) ──────────────────────────────
                # labels[i] = token the model should predict at position i
                # -1 means "ignore this position in loss computation"

                labels = [-1] * len(seq)  # start with all positions masked

                # Unmask only response token positions
                # response starts at position n_prefix-1
                # (because label[i] = seq[i+1], shifted by 1)
                for i in range(n_prefix - 1, len(seq) - 1):
                    labels[i] = seq[i + 1]  # label = next token in sequence

                # Example with n_prefix=3:
                # seq    : [BOS, I1, I2, R1, R2, R3, EOS]
                # labels : [-1,  -1, -1, R2, R3, EOS, -1]
                #           instruction masked  response unmasked

                # Standard LM shift: input=seq[:-1], target=labels[:-1]
                # Drop last token from both (can't predict beyond sequence)
                self.samples.append((seq[:-1], labels[:-1]))

        print(f"  Dataset: {len(self.samples):,} samples | {skipped} skipped (too long / empty)")

    def __len__(self):
        # Returns total number of samples in dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns one (input_ids, labels) pair as PyTorch tensors
        ids, labels = self.samples[idx]
        return (
            torch.tensor(ids,    dtype=torch.long),  # input token IDs
            torch.tensor(labels, dtype=torch.long),  # labels (-1 for masked, token ID for response)
        )


def collate_fn(batch: list, pad_id: int):
    """
    Custom collate function for DataLoader.

    Pads variable-length sequences to the longest sequence in the batch.
    This is needed because different instruction-response pairs have different lengths.

    Padding strategy:
      input_ids : padded with pad_id  (real padding token)
      labels    : padded with -1      (ignored in loss computation)
    """

    xs, ys = zip(*batch)  # separate inputs and labels from batch

    # Find longest sequence in this batch
    max_len = max(x.size(0) for x in xs)

    # Create padded tensors filled with pad values
    x_pad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)  # [B, max_len]
    y_pad = torch.full((len(ys), max_len), -1,     dtype=torch.long)  # [B, max_len]

    # Copy actual data into padded tensors
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, :x.size(0)] = x  # copy input tokens
        y_pad[i, :y.size(0)] = y  # copy labels

    # Remaining positions stay as pad_id and -1 respectively
    return x_pad, y_pad


# ── LR schedule (cosine with warmup) ────────────────────────────────────────

def get_lr(step: int) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.

    Phase 1 (0 → WARMUP_STEPS):    linear increase from 0 to LR
    Phase 2 (WARMUP_STEPS → MAX_STEPS): cosine decay from LR to min_lr
    """

    min_lr = LR * 0.1  # minimum LR = 10% of peak = 6e-6

    # Phase 1: Linear warmup
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
        # step 0   → LR × 1/100  = 6e-7  (near zero)
        # step 50  → LR × 51/100 = 3.06e-5
        # step 99  → LR × 100/100= 6e-5  (peak)

    # Safety: hold at min_lr after MAX_STEPS
    if step >= MAX_STEPS:
        return min_lr

    # Phase 2: Cosine decay
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    # progress = 0.0 at start of decay → 1.0 at end

    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (LR - min_lr)
    # progress=0.0 → returns LR      (peak)
    # progress=0.5 → returns mid LR  (halfway)
    # progress=1.0 → returns min_lr  (minimum)


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def save_ckpt(model, optimizer, step: int, val_loss: float, tag: str):
    """Saves model weights, optimizer state, and metadata to disk."""

    os.makedirs(CKPT_DIR, exist_ok=True)  # create checkpoints_ft/ if not exists

    # Unwrap torch.compile wrapper to get original model
    raw = getattr(model, "_orig_mod", model)
    # model._orig_mod exists if torch.compile was used
    # otherwise returns model unchanged

    path = os.path.join(CKPT_DIR, f"ft_ckpt_{tag}.pt")
    # tag="best"   → checkpoints_ft/ft_ckpt_best.pt
    # tag="latest" → checkpoints_ft/ft_ckpt_latest.pt
    # tag="final"  → checkpoints_ft/ft_ckpt_final.pt

    torch.save({
        "step"     : step,                    # current training step
        "model"    : raw.state_dict(),        # all model weights (31.32M params)
        "optimizer": optimizer.state_dict(),  # AdamW momentum states (m and v)
        "val_loss" : val_loss,                # validation loss at this checkpoint
    }, path)

    print(f"  Saved → {path}  (step={step}  val_loss={val_loss:.4f})")


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()  # decorator: disables gradient computation for entire function
def validate(model, loader: DataLoader, device: str) -> float:
    """
    Computes average validation loss over entire validation set.
    Uses masked loss: only response tokens contribute (same as training).
    """

    model.eval()     # disable dropout → deterministic output
    total, n = 0.0, 0  # accumulator for loss sum and batch count

    for x, y in loader:  # iterate over ALL validation batches (no max_batches limit)
        x, y = x.to(device), y.to(device)  # move tensors to GPU

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # bfloat16 mixed precision → faster, less memory

            logits, _ = model(x)
            # model(x) returns (logits, loss)
            # logits shape: [B, T, vocab_size] = [16, T, 32000]
            # _ discards the internal loss (we compute our own below)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                # reshape logits: [B, T, 32000] → [B*T, 32000]

                y.view(-1),
                # reshape labels: [B, T] → [B*T]

                ignore_index=-1,
                # positions where label=-1 are IGNORED in loss
                # these are instruction tokens and padding tokens
            )

        total += loss.item()  # accumulate loss (.item() converts tensor to float)
        n += 1                # count batches

    model.train()  # re-enable dropout for training
    return total / max(n, 1)  # average loss (max(n,1) prevents division by zero)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():

    # ── Device setup ─────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use GPU if available, otherwise CPU

    torch.manual_seed(42)  # set random seed for reproducibility

    # Enable TF32 for faster matrix operations on Ampere/Ada GPUs (RTX 4060 Ti)
    torch.backends.cuda.matmul.allow_tf32 = True  # speeds up matmul (attention, FFN)
    torch.backends.cudnn.allow_tf32       = True  # speeds up conv operations

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    # loads the BPE tokenizer trained during setup_data.py
    # same tokenizer used for pretraining → consistent vocabulary

    pad_id = tokenizer.token_to_id("<|pad|>")
    # get ID of padding token → used in collate_fn to pad short sequences

    print(f"Tokenizer loaded — vocab_size={tokenizer.get_vocab_size():,}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    cfg = ModelConfig()
    # creates config with all hyperparameters
    # context_len = 256 for fine-tuning0.

    full_ds = InstructionDataset(DATA_PATH, tokenizer, cfg.context_len)
    # tokenises all 3075 instruction-response pairs
    # applies response-only loss masking

    n_val   = max(1, int(len(full_ds) * VAL_RATIO))
    # VAL_RATIO = 0.05 → 5% for validation
    # 3075 × 0.05 = 153 validation samples
    # max(1, ...) ensures at least 1 val sample

    n_train = len(full_ds) - n_val
    # remaining 2922 samples for training

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
        # fixed seed → same train/val split every run
        # reproducible results
    )

    # partial creates a new function with pad_id pre-filled
    # collate_fn needs pad_id but DataLoader only passes batch
    collate = partial(collate_fn, pad_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,       # 16 sequences per batch
        shuffle=True,                # shuffle training data each epoch
        collate_fn=collate,          # pad variable-length sequences
        num_workers=2,               # 2 parallel CPU workers for data loading
        pin_memory=True,             # faster CPU→GPU transfer
        persistent_workers=True,     # keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,       # same batch size as training
        shuffle=False,               # no shuffle for validation (deterministic)
        collate_fn=collate,          # same padding function
        num_workers=2,               # 2 parallel workers
        pin_memory=True,             # faster CPU→GPU transfer
    )

    # ── Model — load pretrained weights ──────────────────────────────────────
    model = GPT(cfg).to(device)
    # creates GPT model with random weights
    # moves all 31.32M parameters to GPU

    if os.path.exists(PRETRAINED_CKPT):
        ckpt  = torch.load(PRETRAINED_CKPT, map_location=device)
        # loads checkpoint dictionary from disk
        # map_location=device ensures weights loaded onto correct device

        state = ckpt.get("model", ckpt)
        # ckpt["model"] contains the model state_dict
        # fallback to ckpt itself if "model" key doesn't exist

        model.load_state_dict(state, strict=True)
        # copies pretrained weights into model
        # strict=True: every key must match exactly
        # before: random weights  after: pretrained weights

        print(f"Loaded pretrained weights <- {PRETRAINED_CKPT}")
    else:
        print(f"WARNING: {PRETRAINED_CKPT} not found — fine-tuning from random init")

    # ── Reduce dropout for fine-tuning ───────────────────────────────────────
    for m in model.modules():
        # iterates over ALL submodules (layers) in the model
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.05
            # reduces dropout from 0.1 (pretraining) to 0.05 (fine-tuning)
            # model is already regularised by pretrained weights
            # less dropout needed → better learning on small dataset

    # ── Optimizer ────────────────────────────────────────────────────────────
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    # weight matrices have dim >= 2 (e.g., [448, 448], [448, 1792])
    # these get weight decay applied → L2 regularisation

    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    # biases and LayerNorm params have dim < 2 (1D vectors)
    # these do NOT get weight decay → standard practice

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": WEIGHT_DECAY},  # 0.1
            {"params": no_decay_params, "weight_decay": 0.0},           # no decay
        ],
        lr=LR,              # peak learning rate = 6e-5
        betas=(0.9, 0.95),  # momentum parameters (same as pretraining)
    )

    # ── torch.compile ─────────────────────────────────────────────────────────
    if hasattr(torch, "compile") and sys.platform != "win32":
        # hasattr check: torch.compile requires PyTorch >= 2.0
        # sys.platform != "win32": torch.compile has issues on Windows

        print("Compiling model with torch.compile() ...")
        model = torch.compile(model)
        # converts model to optimised machine code
        # fuses GPU kernels → ~20% speedup
        # takes 3-4 minutes first time (cached afterwards)

    # ── Print training summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Instruction Fine-tuning")
    print(f"  Device      : {device}")
    print(f"  Train       : {n_train}  |  Val : {n_val}")
    print(f"  Batch       : {BATCH_SIZE} × grad_accum {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} eff.")
    print(f"  LR          : {LR}  |  Steps : {MAX_STEPS}")
    print(f"  Checkpoints : {CKPT_DIR}/")
    print(f"{'='*60}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()            # enable dropout for training
    loader_iter = iter(train_loader)  # create iterator for manual batch fetching
    best_val    = float("inf")        # tracks best validation loss seen so far
    t_start     = time.perf_counter() # record training start time

    for step in range(MAX_STEPS):  # iterate from 0 to 1499

        # ── Update learning rate ──────────────────────────────────────────────
        lr = get_lr(step)  # compute LR for this step using cosine schedule
        for g in optimizer.param_groups:
            g["lr"] = lr   # update LR in both param groups (decay and no_decay)

        # ── Gradient accumulation ─────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        # clears gradients from previous step
        # set_to_none=True: sets grad=None instead of zeros
        # → saves memory, slightly faster

        loss_accum = 0.0  # accumulates loss across micro-steps for logging

        for _ in range(GRAD_ACCUM):  # loop GRAD_ACCUM=2 times per optimizer step
            try:
                x, y = next(loader_iter)  # get next batch from DataLoader
            except StopIteration:
                # DataLoader exhausted → start new epoch
                loader_iter = iter(train_loader)  # reset iterator
                x, y = next(loader_iter)           # get first batch of new epoch

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # move tensors from CPU to GPU
            # non_blocking=True: async transfer → CPU doesn't wait for GPU

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # bfloat16 mixed precision → 2× faster, 2× less memory

                logits, _ = model(x)
                # forward pass through GPT model
                # logits: [B, T, 32000] → probability distribution over vocabulary
                # _: internal loss discarded (we compute masked loss below)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    # reshape: [B, T, 32000] → [B*T, 32000]

                    y.view(-1),
                    # reshape: [B, T] → [B*T]

                    ignore_index=-1,
                    # KEY: positions with label=-1 are IGNORED
                    # instruction tokens → -1 → not counted in loss
                    # response tokens   → real ID → counted in loss
                    # padding tokens    → -1 → not counted in loss
                )

            (loss / GRAD_ACCUM).backward()
            # divide by GRAD_ACCUM to normalise gradients
            # without division: gradients 2× too large
            # .backward(): computes and ACCUMULATES gradients
            # (not overwritten, hence "accumulation")

            loss_accum += loss.item() / GRAD_ACCUM
            # accumulate normalised loss for logging
            # .item(): converts GPU tensor to Python float

        # ── Gradient clipping and optimizer step ─────────────────────────────
        raw = getattr(model, "_orig_mod", model)
        # unwrap torch.compile to access raw model parameters

        torch.nn.utils.clip_grad_norm_(raw.parameters(), GRAD_CLIP)
        # clips gradient norm to maximum 1.0
        # prevents exploding gradients → stable training

        optimizer.step()
        # updates all parameters using accumulated gradients
        # AdamW: param -= lr × m / (√v + ε)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:  # print every 20 steps
            elapsed = (time.perf_counter() - t_start) / 60
            # compute elapsed time in minutes

            print(f"step {step:4d}/{MAX_STEPS} | loss {loss_accum:.4f} | "
                  f"ppl {math.exp(min(loss_accum, 20)):7.2f} | "
                  f"lr {lr:.2e} | elapsed {elapsed:.1f}m")
            # loss_accum: average loss over 2 micro-steps
            # ppl: perplexity = e^loss (min with 20 prevents overflow)
            # lr: current learning rate in scientific notation
            # elapsed: minutes since training started

        # ── Validation and checkpointing ──────────────────────────────────────
        if step > 0 and step % VAL_EVERY == 0:
            # step > 0: skip validation at step 0 (random weights)
            # every 100 steps: run full validation

            val_loss = validate(model, val_loader, device)
            # compute average loss on all 153 validation samples

            print(f"\n  ┌── VALIDATION  step {step} ──────────────────────")
            print(f"  │  val_loss={val_loss:.4f}  ppl={math.exp(min(val_loss, 20)):.2f}")

            save_ckpt(model, optimizer, step, val_loss, "latest")
            # always save latest → used for resuming if training crashes

            if val_loss < best_val:
                best_val = val_loss        # update best val loss seen
                save_ckpt(model, optimizer, step, val_loss, "best")
                # save best → only overwritten when validation improves
                print(f"  │  ✓ New best checkpoint!")

            print(f"  └────────────────────────────────────────────────\n")

    # ── End of training ───────────────────────────────────────────────────────
    save_ckpt(model, optimizer, MAX_STEPS, best_val, "final")
    # save final checkpoint regardless of val loss

    total_min = (time.perf_counter() - t_start) / 60
    # total training time in minutes

    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete!")
    print(f"  Total time  : {total_min:.1f}m")
    print(f"  Best val    : {best_val:.4f}  (ppl {math.exp(min(best_val, 20)):.2f})")
    print(f"  Checkpoints : {CKPT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
    # runs main() only when script is executed directly
    # not when imported as a module