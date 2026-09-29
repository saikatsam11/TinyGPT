"""
Usage:
    # Interactive mode (type prompts in terminal)
    python generate.py --ckpt ckpt_best.pt

    # One-shot
    python generate.py --ckpt ckpt_best.pt --prompt "Once upon a time there was a little girl"

    # Tweak sampling
    python generate.py --ckpt ckpt_best.pt --prompt "The dog ran to" --max_tokens 200 --temperature 0.9 --top_p 0.95
"""
# ^ Module-level docstring showing CLI usage examples for all three run modes

import argparse                      # Parse command-line flags like --ckpt, --prompt, --temperature
import torch                         # PyTorch: used for loading checkpoints, tensor creation, and no_grad inference
from tokenizers import Tokenizer     # HuggingFace fast tokenizer: encodes text to token IDs and decodes back

from Model.config import ModelConfig # Local dataclass that holds every model hyperparameter (layers, heads, dims, etc.)
from Model.gpt    import GPT         # Local GPT class that defines the full transformer architecture


# ─────────────────────────────────────────────────────────────────────────────
# Load model + tokeniser
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: str):
    # Takes the path to a .pt checkpoint file and the target device string ("cpu" or "cuda")

    ckpt = torch.load(ckpt_path, map_location=device)
    # Deserialise the checkpoint dict from disk; map_location moves all tensors to the chosen device immediately

    cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    # Rebuild the ModelConfig dataclass from the config dict stored inside the checkpoint
    # The dict comprehension filters out any keys that no longer exist in ModelConfig,
    # so old checkpoints don't crash if the config schema has changed since training

    model = GPT(cfg).to(device)
    # Instantiate a fresh GPT model using the restored config, then move all parameters to the target device

    model.load_state_dict(ckpt["model"])
    # Copy the saved weight tensors from the checkpoint into the freshly built model

    model.eval()
    # Switch the model to evaluation mode: disables dropout and makes BatchNorm use running stats

    step     = ckpt.get("step", "?")
    # Read the training step number from the checkpoint; fall back to "?" if the key is missing

    val_loss = ckpt.get("val_loss", float("nan"))
    # Read the validation loss recorded at checkpoint time; fall back to NaN if missing

    print(f"Loaded checkpoint: step={step}  val_loss={val_loss:.4f}")
    # Confirm the checkpoint loaded successfully and show its training progress at a glance

    return model, cfg
    # Return the ready-to-use model and its config so the caller can use both


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate(model, tokenizer, prompt: str, cfg: ModelConfig, device: str,
             max_tokens: int = 300, temperature: float = 0.8, top_p: float = 0.9) -> str:
    # Encodes a text prompt, runs autoregressive sampling, and returns the decoded output string
    # max_tokens:  upper bound on how many new tokens the model may produce
    # temperature: controls randomness — lower = more deterministic, higher = more creative
    # top_p:       nucleus sampling cutoff — only sample from the top-p probability mass

    bos_id = tokenizer.token_to_id("<|bos|>")
    # Look up the integer ID for the beginning-of-sequence special token

    eos_id = tokenizer.token_to_id("<|eos|>")
    # Look up the integer ID for the end-of-sequence special token; generation stops when this is emitted

    enc      = tokenizer.encode(prompt)
    # Run the tokenizer on the raw prompt string, producing an Encoding object with an .ids list

    seed_ids = [bos_id] + enc.ids
    # Prepend the BOS token to the prompt token IDs to form the initial context the model sees

    idx      = torch.tensor([seed_ids], device=device)
    # Wrap the seed IDs in a 2-D tensor of shape (1, seq_len) and place it on the target device

    with torch.no_grad():
        # Disable gradient computation for inference — saves memory and speeds up the forward pass
        out = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_id=eos_id,
        )

    token_ids = [t for t in out[0].tolist() if t not in (bos_id, eos_id)]
    # Extract the first (and only) sequence from the batch, convert to a Python list,
    # and strip out both the BOS and EOS special tokens so they don't appear in the text

    text = tokenizer.decode(token_ids)
    # Convert the list of token IDs back into a human-readable string

    return text.strip()
    # Remove any leading/trailing whitespace before returning the final generated text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    # Entry point: sets up the device, loads model + tokenizer, then runs one-shot or interactive mode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use the GPU if CUDA is available, otherwise fall back to CPU

    print(f"Device: {device}\n")
    # Inform the user which compute device will be used

    model, cfg = load_model(args.ckpt, device)
    # Load the GPT model and its config from the checkpoint path provided on the command line

    print(f"Loading tokeniser from: {cfg.tokenizer_path}")
    # Show the path where the tokenizer file will be read from (stored inside the config)

    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    # Deserialise the HuggingFace tokenizer from the JSON file saved during data preparation

    print(f"Vocab size: {tokenizer.get_vocab_size():,}\n")
    # Display the total vocabulary size so the user can verify it matches the model's embedding table

    if args.prompt:
        # ── Single generation ─────────────────────────────────────────────
        # A prompt was provided on the command line, so run one generation and exit

        story = generate(model, tokenizer, args.prompt, cfg, device,
                         args.max_tokens, args.temperature, args.top_p)
        # Call the generate function with all CLI-supplied sampling parameters

        print(f"{'─'*60}")
        # Print a horizontal divider line for readability

        print(f"Prompt : {args.prompt}")
        # Echo the original prompt so the user can see input vs output clearly

        print(f"{'─'*60}")
        # Print another divider to separate the prompt label from the generated text

        print(story)
        # Print the generated story text

        print(f"{'─'*60}")
        # Print a final divider to mark the end of the output block

    else:
        # ── Interactive mode ──────────────────────────────────────────────
        # No prompt was given, so enter a REPL loop that accepts prompts from stdin

        temperature = args.temperature
        # Initialise the mutable temperature variable from the CLI default (can be changed mid-session)

        top_p       = args.top_p
        # Initialise the mutable top_p variable from the CLI default (can be changed mid-session)

        max_tokens  = args.max_tokens
        # Initialise the mutable max_tokens variable from the CLI default (can be changed mid-session)

        print("Interactive mode — type a story prompt and press Enter.")
        # Tell the user they are in interactive mode

        print("Commands:  :temp <float>  |  :top_p <float>  |  :len <int>  |  :quit")
        # List the special colon-prefixed commands that adjust settings without generating text

        print(f"Current settings: temp={temperature}  top_p={top_p}  max_tokens={max_tokens}\n")
        # Show the active sampling settings before the first prompt

        while True:
            # Loop indefinitely until the user types :quit or sends EOF / Ctrl-C

            try:
                prompt = input("Prompt> ").strip()
                # Read a line from stdin, strip surrounding whitespace, store as prompt

            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl-D (EOF) or Ctrl-C gracefully instead of crashing
                print("\nBye!"); break
                # Print a farewell message and exit the loop

            if not prompt:
                continue
                # Skip empty lines — just re-display the prompt without generating

            if prompt == ":quit":
                break
                # Exit the loop cleanly when the user explicitly types :quit

            if prompt.startswith(":temp "):
                temperature = float(prompt.split()[1])
                # Parse the new temperature value from ":temp <float>" and update the variable
                print(f"temperature = {temperature}"); continue
                # Confirm the change and skip to the next iteration without generating

            if prompt.startswith(":top_p "):
                top_p = float(prompt.split()[1])
                # Parse the new top_p value from ":top_p <float>" and update the variable
                print(f"top_p = {top_p}"); continue
                # Confirm the change and skip to the next iteration without generating

            if prompt.startswith(":len "):
                max_tokens = int(prompt.split()[1])
                # Parse the new max_tokens value from ":len <int>" and update the variable
                print(f"max_tokens = {max_tokens}"); continue
                # Confirm the change and skip to the next iteration without generating

            story = generate(model, tokenizer, prompt, cfg, device,
                             max_tokens, temperature, top_p)
            # Generate a story from the user's prompt using the current sampling settings

            print(f"\n{'─'*60}")
            # Print a blank line followed by a divider before the output

            print(story)
            # Print the generated story text

            print(f"{'─'*60}\n")
            # Print a closing divider followed by a blank line after the output


if __name__ == "__main__":
    # Only run the following block when this script is executed directly, not when imported as a module

    p = argparse.ArgumentParser(description="Generate stories from a trained GPT checkpoint")
    # Create the argument parser with a helpful description shown in --help output

    p.add_argument("--ckpt",        required=True,          help="Path to .pt checkpoint")
    # Required positional-style flag: the path to the saved model checkpoint file

    p.add_argument("--prompt",      default=None,           help="Prompt string (omit for interactive)")
    # Optional prompt string; if omitted the script enters interactive REPL mode

    p.add_argument("--max_tokens",  type=int,   default=300,help="Max tokens to generate")
    # Maximum number of new tokens to generate; defaults to 300

    p.add_argument("--temperature", type=float, default=0.8,help="Sampling temperature")
    # Sampling temperature controlling output randomness; defaults to 0.8

    p.add_argument("--top_p",       type=float, default=0.9,help="Nucleus sampling top-p")
    # Nucleus sampling probability mass cutoff; defaults to 0.9

    main(p.parse_args())
    # Parse all provided command-line arguments and pass the resulting Namespace object to main()