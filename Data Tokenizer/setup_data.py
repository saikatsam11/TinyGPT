"""
setup_data.py
──────────────
Run this ONCE before training. It:
  1. Loads 75% of TinyStories from HuggingFace (+ 5% held-out val shard)
  2. Trains a BPE tokeniser (vocab=32K) and saves it
  3. Tokenises all stories → packed .bin shards ready for training
     Each story is wrapped as:  [BOS] [tokens...] [EOS]

Usage:
    python setup_data.py

Outputs:
    tokenizer/tokenizer.json        ← saved tokeniser (needed at inference too)
    data/bin/train_shard_XXXX.bin   ← uint16 token ID train shards
    data/bin/val_shard_XXXX.bin     ← uint16 token ID val shards
"""

import os
import numpy as np
from datasets   import load_dataset
from tokenizers import Tokenizer
from tokenizers.models          import BPE
from tokenizers.trainers        import BpeTrainer
from tokenizers.pre_tokenizers  import ByteLevel
from tokenizers.normalizers     import NFC
from tokenizers.decoders        import ByteLevel as ByteLevelDecoder   # FIX 1: add decoder
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_SIZE     = 32_000
# Total number of tokens in the learned BPE vocabulary, including special tokens

SHARD_SIZE     = 10_000_000
# Maximum number of tokens stored per binary shard file (~20 MB each at 2 bytes/token for uint16)

TOKENIZER_OUT  = "tokenizer/tokenizer.json"
# File path where the trained tokenizer will be saved as a JSON file

DATA_OUT_DIR   = "data/bin"
# Directory where all binary shard files (.bin) for train and val splits will be written

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
# List of special tokens added to the vocabulary before BPE training
# Order is intentional: pad=ID 0, bos=ID 1, eos=ID 2, unk=ID 3 (FIX 2: pad first so ID 0 is the null token)
#                   ID 0        ID 1       ID 2       ID 3


# ── Step 1: Load dataset ──────────────────────────────────────────────────────
print("=" * 60)
# Print a visual section separator (60 equal signs) for readability in the terminal

print("STEP 1 — Loading TinyStories (75% train + 5% val)")
# Announce that step 1 (dataset loading) is beginning

print("=" * 60)
# Print a closing separator to frame the step header

ds      = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
# Download (or load from cache) the full TinyStories "train" split from HuggingFace Hub

n_total = len(ds)
# Count the total number of stories in the downloaded split

n_train = int(0.75 * n_total)
# Compute how many stories to use for training: 75% of the full dataset

n_val   = int(0.05 * n_total)
# Compute how many stories to hold out for validation: 5% of the full dataset (FIX 3: dedicated val split)

train_texts = ds.select(range(n_train))["text"]
# Select the first n_train stories and extract just their "text" column as a list of strings

val_texts   = ds.select(range(n_train, n_train + n_val))["text"]
# Select the next n_val stories immediately after the train split and extract their text

print(f"  Train stories : {len(train_texts):,}")
# Print the exact number of training stories with thousands-separator formatting

print(f"  Val   stories : {len(val_texts):,}")
# Print the exact number of validation stories with thousands-separator formatting


# ── Step 2: Train BPE tokeniser ───────────────────────────────────────────────
print("\n" + "=" * 60)
# Print a blank line followed by a section separator before step 2

print("STEP 2 — Training BPE tokeniser")
# Announce that step 2 (tokenizer training) is beginning

print("=" * 60)
# Print a closing separator to frame the step header

os.makedirs(os.path.dirname(TOKENIZER_OUT), exist_ok=True)
# Create the "tokenizer/" directory if it doesn't already exist; exist_ok=True prevents errors if it does

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
# Instantiate a new Tokenizer wrapping a BPE model; unknown tokens will map to the <|unk|> special token

tokenizer.normalizer    = NFC()
# Attach the NFC Unicode normalizer so all input text is canonically composed before tokenization

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
# Attach the byte-level pre-tokenizer; add_prefix_space=True ensures consistent tokenization at word boundaries

tokenizer.decoder       = ByteLevelDecoder()
# Attach the byte-level decoder so token IDs can be correctly converted back to the original text (FIX 1)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,      # Target vocabulary size including special tokens
    special_tokens=SPECIAL_TOKENS,  # Reserve these tokens with guaranteed low IDs before BPE merges
    min_frequency=2,            # A byte-pair merge is only learned if it appears at least twice in the corpus
    show_progress=True,         # Display a progress bar while the BPE algorithm runs
)
# Build the BpeTrainer configuration object that will guide the vocabulary learning process

tokenizer.train_from_iterator(train_texts, trainer=trainer, length=len(train_texts))
# Run BPE training over the train_texts iterator; length hint lets tqdm show accurate progress
# Only train split is used here — val texts are never seen by the tokenizer (no data leakage)

tokenizer.save(TOKENIZER_OUT)
# Serialise the fully trained tokenizer (model, normalizer, pre-tokenizer, decoder) to a JSON file

print(f"\n  Tokeniser saved → {TOKENIZER_OUT}")
# Confirm the tokenizer file was written and show its path

print(f"  Actual vocab size : {tokenizer.get_vocab_size():,}")
# Print the real vocab size (may differ slightly from VOCAB_SIZE due to merge constraints)

sample = tokenizer.encode("Once upon a time there was a little girl.")
# Encode a known sentence to verify the tokenizer works correctly after saving

print(f"  Sample tokens : {sample.tokens}")
# Print the human-readable subword tokens for the sample sentence

PAD_ID = tokenizer.token_to_id("<|pad|>")
# Look up the integer ID assigned to the padding special token (should be 0)

BOS_ID = tokenizer.token_to_id("<|bos|>")
# Look up the integer ID assigned to the beginning-of-sequence special token (should be 1)

EOS_ID = tokenizer.token_to_id("<|eos|>")
# Look up the integer ID assigned to the end-of-sequence special token (should be 2)

UNK_ID = tokenizer.token_to_id("<|unk|>")
# Look up the integer ID assigned to the unknown token (should be 3)

print(f"  PAD={PAD_ID}  BOS={BOS_ID}  EOS={EOS_ID}  UNK={UNK_ID}")
# Print all four special token IDs to verify they were assigned in the intended order

decoded = tokenizer.decode(sample.ids)
# Decode the sample token IDs back to a string to verify the round-trip is lossless

print(f"  Decoded       : {decoded}")
# Print the decoded string; it should match (or closely match) the original sample sentence


# ── Step 3: Tokenise → .bin shards ───────────────────────────────────────────
print("\n" + "=" * 60)
# Print a blank line followed by a section separator before step 3

print("STEP 3 — Tokenising stories → .bin shards")
# Announce that step 3 (bulk tokenization and shard writing) is beginning

print("=" * 60)
# Print a closing separator to frame the step header

os.makedirs(DATA_OUT_DIR, exist_ok=True)
# Create the "data/bin/" output directory if it doesn't already exist


def flush_shard(buf: list, idx: int, prefix: str) -> tuple[int, list]:
    # Helper that writes the current token buffer to a binary shard file and resets it
    # buf:    list of integer token IDs accumulated so far
    # idx:    current shard index used to build a zero-padded filename
    # prefix: "train" or "val" — determines the filename prefix
    # returns (next_shard_index, empty_list) so the caller can reset both counters

    arr  = np.array(buf, dtype=np.uint16)
    # Convert the Python list of token IDs to a NumPy array of unsigned 16-bit integers (max ID 65535)

    path = os.path.join(DATA_OUT_DIR, f"{prefix}_shard_{idx:04d}.bin")
    # Build the full output path, e.g. "data/bin/train_shard_0000.bin"; :04d zero-pads the index to 4 digits

    arr.tofile(path)
    # Write the raw binary bytes of the uint16 array directly to disk (no header, no metadata)

    print(f"  [{prefix}] Shard {idx:04d}: {len(arr):,} tokens → {path}")
    # Log the shard that was just written: its prefix, index, token count, and file path

    return idx + 1, []
    # Return the incremented shard index and a fresh empty list for the next shard's buffer


def tokenise_and_shard(texts: list[str], prefix: str) -> dict:
    """Encode texts into packed BOS+tokens+EOS shards."""
    # Takes a list of raw story strings and a split prefix ("train" or "val")
    # Encodes every story, wraps it with BOS/EOS, packs tokens into fixed-size shards, and flushes to disk
    # Returns a summary dict with the total shard count and total token count

    shard_idx    = 0
    # Index of the next shard file to be written; incremented after every flush

    token_buf    = []
    # Accumulator list that collects token IDs until a full shard is ready to flush

    total_tokens = 0
    # Running count of all tokens processed across all shards for this split

    BATCH        = 5_000
    # Number of stories to encode in one encode_batch call; balances memory use and throughput

    for i in tqdm(range(0, len(texts), BATCH), desc=f"Encoding [{prefix}]"):
        # Iterate over story indices in steps of BATCH; tqdm shows a live progress bar

        batch     = texts[i : i + BATCH]
        # Slice out the next batch of up to BATCH raw story strings

        encodings = tokenizer.encode_batch(batch)
        # Encode all stories in the batch in parallel; returns a list of Encoding objects

        for enc in encodings:
            # Iterate over each individual story's encoding result

            story_tokens = [BOS_ID] + enc.ids + [EOS_ID]
            # Wrap the story's token IDs with BOS at the start and EOS at the end (FIX 4)
            # This lets the model learn to start and stop stories during training

            token_buf.extend(story_tokens)
            # Append this story's tokens to the running buffer

            total_tokens += len(story_tokens)
            # Add this story's token count (including BOS and EOS) to the running total

            if len(token_buf) >= SHARD_SIZE:
                # Check if the buffer has reached or exceeded the shard size threshold

                shard_idx, token_buf = flush_shard(token_buf, shard_idx, prefix)
                # Write the full buffer to disk and reset it; update the shard index

    if token_buf:
        # After the loop, check if any tokens remain in the buffer that haven't been flushed yet

        shard_idx, _ = flush_shard(token_buf, shard_idx, prefix)
        # Flush the final partial shard to disk; discard the returned empty buffer with _

    return {"shards": shard_idx, "tokens": total_tokens}
    # Return a summary dict: total number of shards written and total tokens encoded for this split


print("\n── Train shards ──")
# Print a sub-header to visually separate the train shard generation output

train_stats = tokenise_and_shard(train_texts, prefix="train")
# Tokenise all training stories and write them to train_shard_XXXX.bin files; store the summary

print("\n── Val shards ──")
# Print a sub-header to visually separate the validation shard generation output

val_stats   = tokenise_and_shard(val_texts,   prefix="val")
# Tokenise all validation stories and write them to val_shard_XXXX.bin files; store the summary


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
# Print a blank line followed by a final section separator

print("DONE")
# Announce that all three steps have completed successfully

print("=" * 60)
# Print a closing separator to frame the DONE header

print(f"  Train shards : {train_stats['shards']}   "
      f"({train_stats['tokens']/1e6:.1f}M tokens)")
# Print the total number of train shard files written and the total train token count in millions

print(f"  Val   shards : {val_stats['shards']}   "
      f"({val_stats['tokens']/1e6:.1f}M tokens)")
# Print the total number of val shard files written and the total val token count in millions

print(f"  Data dir     : {DATA_OUT_DIR}/")
# Remind the user where all the binary shard files were saved

print(f"  Tokeniser    : {TOKENIZER_OUT}")
# Remind the user where the trained tokenizer JSON file was saved

print(f"\nAll set! Run  python train.py  to start pretraining.")
# Final instruction telling the user the next step now that data preparation is completeSonnet 4.6