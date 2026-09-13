import os, math, time, glob
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel  import DistributedDataParallel as DDP
from torch.utils.data   import Dataset, DataLoader, DistributedSampler

from Model.config import ModelConfig
from Model.gpt    import GPT


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ShardedDataset(Dataset):
    def __init__(self, data_dir, context_len, split="train"):
        shards = sorted(glob.glob(os.path.join(data_dir, f"{split}_shard_*.bin")))
        assert shards, f"No {split} shards found in {data_dir}"

        self.context_len = context_len
        self.mmaps   = [np.memmap(s, dtype=np.uint16, mode="r") for s in shards] # memory-map the shards (fast, doesn't load into RAM)
        self.lengths = [max(0, len(m) - context_len) for m in self.mmaps] # number of full context windows in each shard
        self.cumlen  = np.cumsum([0] + self.lengths) # cumulative lengths to find shard boundaries
        self.total   = int(self.cumlen[-1]) # total number of windows across all shards

        print(f"  {split.upper()} Dataset: {len(shards)} shards | "
              f"{sum(len(m) for m in self.mmaps):,} tokens | "
              f"{self.total:,} windows")

    def __len__(self): return self.total # total number of context windows across all shards

    def __getitem__(self, idx):
        shard = int(np.searchsorted(self.cumlen[1:], idx, side="right")) # find which shard the index falls into
        local = idx - int(self.cumlen[shard]) # index within the shard
        chunk = self.mmaps[shard][local : local + self.context_len + 1] # +1 for target token
        chunk = torch.from_numpy(chunk.astype(np.int64)) # convert to PyTorch tensor
        return chunk[:-1], chunk[1:] # input tokens and target tokens (next token prediction)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(step, cfg):
    min_lr = cfg.lr * 0.1
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_ckpt(model, optimizer, step, val_loss, cfg, tag):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    raw  = model.module if isinstance(model, DDP) else model
    # If compiled, get the original module
    raw  = getattr(raw, "_orig_mod", raw)
    path = os.path.join(cfg.ckpt_dir, f"ckpt_{tag}.pt")
    torch.save({"step": step, "model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss, "config": cfg.__dict__}, path)
    print(f"  Saved → {path}  (step={step}  val_loss={val_loss:.4f})")


def load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    raw  = model.module if isinstance(model, DDP) else model
    raw  = getattr(raw, "_orig_mod", raw)
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed ← {path}  (step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f})")
    return ckpt["step"], ckpt["val_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, max_batches=100):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.item(); n += 1
    model.train()
    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = ModelConfig()

    # ── DDP ──────────────────────────────────────────────────────────────────
    ddp        = int(os.environ.get("RANK", -1)) != -1
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)

    master = (rank == 0)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42 + rank)

    # ── 4060 Ti specific: enable TF32 for matrix ops ──────────────────────────
    # TF32 keeps full range but reduces precision slightly — big speedup on Ada
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if master:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        eff_tokens = cfg.batch_size * cfg.grad_accum * cfg.context_len * world_size
        print(f"\n{'='*60}")
        print(f"  RTX 4060 Ti Training Run")
        print(f"{'='*60}")
        print(f"  Device        : {device}")
        print(f"  Batch size    : {cfg.batch_size} seqs/step")
        print(f"  Grad accum    : {cfg.grad_accum}  →  effective {cfg.batch_size * cfg.grad_accum} seqs")
        print(f"  Tokens/step   : {eff_tokens:,}")
        print(f"  Max steps     : {cfg.max_steps:,}  (~2 epochs)")
        print(f"  Total tokens  : {eff_tokens * cfg.max_steps / 1e6:.0f}M")
        print(f"  torch.compile : {cfg.compile_model}")
        print(f"{'='*60}\n")

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
        train_dataset = ShardedDataset(cfg.data_dir, cfg.context_len, split="train") # create training dataset from sharded binary files (memory-mapped for efficiency)
        val_dataset   = ShardedDataset(cfg.data_dir, cfg.context_len, split="val")

        train_sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True) if ddp else None

        train_loader = DataLoader( # Batches and parallelizes data loading to feed data efficiently to GPU during training.
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        ) 

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
    # ── Model ─────────────────────────────────────────────────────────────────
    model     = GPT(cfg).to(device)
    optimizer = model.configure_optimizer(cfg)

    # torch.compile — significant speedup on 4060 Ti (Ada Lovelace, sm_89)
    if cfg.compile_model and hasattr(torch, "compile"):
        if master: print("  Compiling model with torch.compile() ...")
        model = torch.compile(model)
        if master: print("  Compilation done.\n")

    if ddp:
        model = DDP(model, device_ids=[rank])

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step, best_val = 0, float("inf")
    latest = os.path.join(cfg.ckpt_dir, "ckpt_latest.pt")
    if os.path.exists(latest):
        start_step, best_val = load_ckpt(latest, model, optimizer, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    loader_iter = iter(train_loader)
    step_times  = []
    t_start     = time.perf_counter()

    for step in range(start_step, cfg.max_steps):
        t0 = time.perf_counter()

        # Update LR
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True) # reset gradients to zero (set_to_none=True is more efficient than zeroing out tensors)
        loss_accum = 0.0

        for micro in range(cfg.grad_accum):
            try:
                x, y = next(loader_iter) # get next batch of data; if we exhaust the DataLoader, we catch the StopIteration exception to reset the iterator and start a new epoch
            except StopIteration:
                if ddp and sampler: sampler.set_epoch(step)
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # move data to GPU asynchronously (non_blocking=True allows overlapping data transfer with computation)

            # Sync grads only on last micro-step (DDP optimization)
            sync_ctx = model.no_sync() if (ddp and micro < cfg.grad_accum - 1) \
                       else torch.cuda.amp.autocast(dtype=torch.bfloat16)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)

            (loss / cfg.grad_accum).backward()
            loss_accum += loss.item() / cfg.grad_accum

        # Gradient clip + step
        raw = model.module if ddp else model
        raw = getattr(raw, "_orig_mod", raw)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.grad_clip)
        optimizer.step()

        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        # ── Logging ───────────────────────────────────────────────────────────
        if master and step % cfg.log_every == 0:
            dt      = step_times[-1]
            tok_s   = cfg.batch_size * cfg.grad_accum * cfg.context_len / dt
            ppl     = math.exp(min(loss_accum, 20))
            elapsed = (t1 - t_start) / 60
            remain  = (cfg.max_steps - step) * sum(step_times[-20:]) / max(len(step_times[-20:]), 1) / 60
            print(f"step {step:5d}/{cfg.max_steps} | "
                  f"loss {loss_accum:.4f} | ppl {ppl:7.2f} | "
                  f"lr {lr:.2e} | "
                  f"{dt*1000:5.0f}ms | {tok_s:6,.0f} tok/s | "
                  f"elapsed {elapsed:.0f}m | eta {remain:.0f}m")

        # ── Validation + checkpoint ────────────────────────────────────────────
        if master and step > 0 and step % cfg.val_every == 0:
            val_loss = validate(model, val_loader, device)
            val_ppl  = math.exp(min(val_loss, 20))
            print(f"\n  ┌── VALIDATION  step {step} ──────────────────────")
            print(f"  │  val_loss = {val_loss:.4f}  |  val_ppl = {val_ppl:.2f}")

            save_ckpt(model, optimizer, step, val_loss, cfg, "latest")

            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(model, optimizer, step, val_loss, cfg, "best")
                print(f"  │  ✓ New best checkpoint!")
            print(f"  └────────────────────────────────────────────────\n")

    # ── End of training ───────────────────────────────────────────────────────
    if master:
        save_ckpt(model, optimizer, cfg.max_steps, best_val, cfg, "final")
        total_time = (time.perf_counter() - t_start) / 3600
        avg_tok_s  = cfg.batch_size * cfg.grad_accum * cfg.context_len / (
                     sum(step_times) / len(step_times))
        print(f"\n{'='*60}")
        print(f"  Pretraining complete!")
        print(f"  Total time    : {total_time:.2f}h")
        print(f"  Avg tok/s     : {avg_tok_s:,.0f}")
        print(f"  Best val loss : {best_val:.4f}  (ppl {math.exp(min(best_val,20)):.2f})")
        print(f"  Checkpoints   : {cfg.ckpt_dir}/")
        print(f"{'='*60}\n")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
