"""
Usage:
    # Interactive mode (type prompts in terminal)
    python generate.py --ckpt ckpt_best.pt

    # One-shot
    python generate.py --ckpt ckpt_best.pt --prompt "Once upon a time there was a little girl"

    # Tweak sampling
    python generate.py --ckpt ckpt_best.pt --prompt "The dog ran to" --max_tokens 200 --temperature 0.9 --top_p 0.95
"""

import argparse
import torch
from tokenizers import Tokenizer

from Model.config import ModelConfig
from Model.gpt    import GPT


# ─────────────────────────────────────────────────────────────────────────────
# Load model + tokeniser
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    # Rebuild config from checkpoint (safe against future config changes)
    cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in ModelConfig.__dataclass_fields__})

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step     = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint: step={step}  val_loss={val_loss:.4f}")
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate(model, tokenizer, prompt: str, cfg: ModelConfig, device: str,
             max_tokens: int = 300, temperature: float = 0.8, top_p: float = 0.9) -> str:

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    enc      = tokenizer.encode(prompt)
    seed_ids = [bos_id] + enc.ids
    idx      = torch.tensor([seed_ids], device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=0.95,
            top_p=0.92,
            eos_id=eos_id,
        )

    token_ids = [t for t in out[0].tolist() if t not in (bos_id, eos_id)]

    text = tokenizer.decode(token_ids)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load model
    model, cfg = load_model(args.ckpt, device)

    # Load tokeniser — saved during setup_data.py
    print(f"Loading tokeniser from: {cfg.tokenizer_path}")
    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    print(f"Vocab size: {tokenizer.get_vocab_size():,}\n")

    if args.prompt:
        # ── Single generation ─────────────────────────────────────────────
        story = generate(model, tokenizer, args.prompt, cfg, device,
                         args.max_tokens, args.temperature, args.top_p)
        print(f"{'─'*60}")
        print(f"Prompt : {args.prompt}")
        print(f"{'─'*60}")
        print(story)
        print(f"{'─'*60}")

    else:
        # ── Interactive mode ──────────────────────────────────────────────
        temperature = args.temperature
        top_p       = args.top_p
        max_tokens  = args.max_tokens

        print("Interactive mode — type a story prompt and press Enter.")
        print("Commands:  :temp <float>  |  :top_p <float>  |  :len <int>  |  :quit")
        print(f"Current settings: temp={temperature}  top_p={top_p}  max_tokens={max_tokens}\n")

        while True:
            try:
                prompt = input("Prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!"); break

            if not prompt:
                continue
            if prompt == ":quit":
                break
            if prompt.startswith(":temp "):
                temperature = float(prompt.split()[1])
                print(f"temperature = {temperature}"); continue
            if prompt.startswith(":top_p "):
                top_p = float(prompt.split()[1])
                print(f"top_p = {top_p}"); continue
            if prompt.startswith(":len "):
                max_tokens = int(prompt.split()[1])
                print(f"max_tokens = {max_tokens}"); continue

            story = generate(model, tokenizer, prompt, cfg, device,
                             max_tokens, temperature, top_p)
            print(f"\n{'─'*60}")
            print(story)
            print(f"{'─'*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate stories from a trained GPT checkpoint")
    p.add_argument("--ckpt",        required=True,          help="Path to .pt checkpoint")
    p.add_argument("--prompt",      default=None,           help="Prompt string (omit for interactive)")
    p.add_argument("--max_tokens",  type=int,   default=300,help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,help="Sampling temperature")
    p.add_argument("--top_p",       type=float, default=0.9,help="Nucleus sampling top-p")
    main(p.parse_args())
