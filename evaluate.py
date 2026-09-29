"""
Evaluates the trained GPT model on N generated stories vs real TinyStories.

Metrics computed:
    - Perplexity (PPL)
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - ROUGE-1, ROUGE-2, ROUGE-L
    - BERTScore (Precision, Recall, F1)
    - MAUVE
    - Distinct-1, Distinct-2
    - Repetition rate (4-gram)
    - Average story length (tokens)
    - Vocabulary coverage

Usage:
    python evaluate.py --ckpt_best.pt --n_stories 500

"""

import os
import re
import csv
import math
import argparse
import torch
import numpy as np
from collections        import Counter
from tokenizers         import Tokenizer
from datasets           import load_dataset
from nltk.translate.bleu_score  import corpus_bleu, SmoothingFunction
from rouge_score        import rouge_scorer as rouge_scorer_lib

from Model.config import ModelConfig
from Model.gpt    import GPT


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    """Loads GPT model from checkpoint and sets to evaluation mode."""
 
    ckpt = torch.load(ckpt_path, map_location=device)
    # loads checkpoint dictionary from disk
    # map_location=device: loads weights onto correct device (CPU or GPU)
 
    cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    # rebuilds ModelConfig from saved config dictionary
    # ckpt["config"] = {d_model: 448, n_layers: 7, ...}
    # only keeps keys that exist in ModelConfig (ignores unknown keys)
    # this is safe if config was updated after checkpoint was saved
 
    model = GPT(cfg).to(device)
    # creates GPT model with correct architecture
    # .to(device): moves all parameters to GPU or CPU
 
    model.load_state_dict(ckpt["model"])
    # loads saved weights into model
    # Before: random weights  After: trained checkpoint weights
 
    model.eval()
    # switches to evaluation mode:
    # disables dropout → deterministic, stable output
    # essential for consistent generation and metric computation
 
    step     = ckpt.get("step", "?")
    # training step at which checkpoint was saved
    # .get("step", "?") returns "?" if key doesn't exist (safe fallback)
 
    val_loss = ckpt.get("val_loss", float("nan"))
    # validation loss at checkpoint time
    # float("nan") fallback if val_loss wasn't saved
 
    print(f"  Loaded checkpoint: step={step}  val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.2f}")
    # math.exp(val_loss) = perplexity from checkpoint
    # e.g. val_loss=1.4830 → ppl=4.41
 
    return model, cfg
    # returns both model and config (cfg needed for tokenizer path etc.)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Generate one story
# ─────────────────────────────────────────────────────────────────────────────
 
@torch.no_grad()
# decorator: disables gradient computation
# generation is inference only → no gradients needed → saves memory
 
def generate_story(model, tokenizer, prompt, cfg, device,
                   max_tokens=200, temperature=0.8, top_p=0.9):
    """
    Generates one story from a text prompt using nucleus (top-p) sampling.
 
    Returns:
        text     : decoded story as a clean string
        token_ids: list of token IDs for the generated story
    """
 
    bos_id = tokenizer.token_to_id("<|bos|>")
    # ID of beginning-of-sequence token (e.g. 1)
    # prepended to every sequence before generation
 
    eos_id = tokenizer.token_to_id("<|eos|>")
    # ID of end-of-sequence token (e.g. 2)
    # generation stops when this token is produced
 
    enc = tokenizer.encode(prompt)
    # tokenises the prompt text into token IDs
    # e.g. "Once upon a time" → [423, 891, 12, 654]
 
    seed_ids = [bos_id] + enc.ids
    # prepend BOS to prompt token IDs
    # e.g. [1, 423, 891, 12, 654]
 
    idx = torch.tensor([seed_ids], device=device)
    # convert to PyTorch tensor and add batch dimension
    # shape: [1, T] where T = len(seed_ids)
    # batch size = 1 (one story at a time)
 
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_p=top_p, eos_id=eos_id)
    # calls GPT.generate() which autoregressively generates tokens
    # max_tokens=200: generate up to 200 new tokens
    # temperature=0.8: controls randomness (lower=safer, higher=creative)
    # top_p=0.9: nucleus sampling (only consider top 90% probability mass)
    # eos_id: stop early if EOS token is generated
    # returns: [1, T + generated_tokens]
 
    token_ids = [t for t in out[0].tolist() if t not in (bos_id, eos_id)]
    # out[0]: removes batch dimension → flat list of token IDs
    # .tolist(): converts tensor to Python list
    # filters out BOS and EOS tokens for clean output
    # these are structural tokens not part of the actual story
 
    text = tokenizer.decode(token_ids)
    # converts token IDs back to text string
    # e.g. [423, 891, 12] → "Once upon a time"
 
    text = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
    # cleans ByteLevel BPE artifacts:
    # Ġ = space before a word (ByteLevel BPE adds this)
    # Ċ = newline character
    # .strip(): removes leading/trailing whitespace
 
    return text, token_ids
    # returns both the decoded text and token IDs
    # token_ids needed for repetition rate and vocab coverage metrics
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
 
def tokenize_words(text):
    """Simple word tokenisation for metric computation."""
 
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())
    # \b: word boundary
    # [a-zA-Z']+: one or more letters or apostrophes (handles "don't", "it's")
    # .lower(): lowercase for case-insensitive comparison
    # extracts clean word tokens for BLEU/ROUGE computation
    # e.g. "Once, Tom ran!" → ["once", "tom", "ran"]
 
 
def compute_bleu(generated_texts, reference_texts):
    """
    Computes corpus-level BLEU-1, BLEU-2, BLEU-3, BLEU-4.
 
    BLEU measures n-gram precision:
    how many n-grams in generated text appear in reference text.
 
    Corpus-level = computed across all stories together (more stable than per-story).
    """
 
    smoothie = SmoothingFunction().method1
    # smoothing prevents BLEU=0 when no n-gram matches found
    # method1: adds small epsilon count to each n-gram order
    # important for short texts or rare n-grams
 
    refs, hyps = [], []
    # refs: list of reference token lists (one per generated story)
    # hyps: list of hypothesis (generated) token lists
 
    for gen, ref in zip(generated_texts, reference_texts):
        hyps.append(tokenize_words(gen))
        # tokenise generated story into word list
        # e.g. ["once", "upon", "a", "time", "there", ...]
 
        refs.append([tokenize_words(ref)])
        # tokenise reference story into word list
        # wrapped in extra list: BLEU supports multiple references per hypothesis
        # we have one reference → [[ref_words]]
 
    scores = {}
    # stores BLEU-1, BLEU-2, BLEU-3, BLEU-4
 
    for n in range(1, 5):
        # n=1: unigrams (individual words)
        # n=2: bigrams  (word pairs)
        # n=3: trigrams (word triples)
        # n=4: 4-grams  (word quadruples)
 
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        # creates weight tuple for nth order BLEU
        # BLEU-1: (1.0, 0.0, 0.0, 0.0) → only unigrams
        # BLEU-2: (0.5, 0.5, 0.0, 0.0) → unigrams + bigrams equally
        # BLEU-3: (0.33, 0.33, 0.33, 0.0)
        # BLEU-4: (0.25, 0.25, 0.25, 0.25) → all 4 orders equally
 
        scores[f"BLEU-{n}"] = corpus_bleu(refs, hyps,
                                           weights=weights,
                                           smoothing_function=smoothie)
        # corpus_bleu: computes BLEU across all story pairs together
        # more reliable than averaging per-story BLEU scores
 
    return scores
    # returns {"BLEU-1": 0.33, "BLEU-2": 0.18, "BLEU-3": 0.12, "BLEU-4": 0.09}
 
 
def compute_rouge(generated_texts, reference_texts):
    """
    Computes ROUGE-1, ROUGE-2, ROUGE-L averaged across all story pairs.
 
    ROUGE measures recall-based n-gram overlap:
    how many reference n-grams appear in generated text.
    """
 
    scorer = rouge_scorer_lib.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
        # use_stemmer=True: reduces words to their root form
        # e.g. "running" → "run", "plays" → "play"
        # increases recall by matching morphological variants
    )
 
    r1, r2, rl = [], [], []
    # accumulate per-story scores for each ROUGE variant
 
    for gen, ref in zip(generated_texts, reference_texts):
        s = scorer.score(ref, gen)
        # computes ROUGE score for this (reference, generated) pair
        # note: ref first, gen second (scorer expects this order)
 
        r1.append(s["rouge1"].fmeasure)
        # rouge1.fmeasure: F1 score combining precision and recall for unigrams
        # F1 = 2 × (precision × recall) / (precision + recall)
 
        r2.append(s["rouge2"].fmeasure)
        # F1 score for bigram overlap
 
        rl.append(s["rougeL"].fmeasure)
        # F1 score for Longest Common Subsequence (LCS)
        # captures sentence-level structure similarity
 
    return {
        "ROUGE-1": np.mean(r1),   # average ROUGE-1 F1 across all stories
        "ROUGE-2": np.mean(r2),   # average ROUGE-2 F1 across all stories
        "ROUGE-L": np.mean(rl),   # average ROUGE-L F1 across all stories
    }
 
 
def compute_bertscore(generated_texts, reference_texts, device):
    """
    Computes BERTScore Precision, Recall, F1.
 
    BERTScore uses contextual embeddings from a pretrained BERT model
    to measure semantic similarity between generated and reference texts.
    Unlike BLEU/ROUGE, captures meaning rather than surface overlap.
 
    rescale_with_baseline=False: returns raw cosine similarity scores
    (0.85-0.95 range for English text)
    """
 
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError(
            "bert-score is not installed. Run:  pip install bert-score"
        )
    # imports bert_score inside function to avoid crash if not installed
    # raises clear error with installation instructions
 
    print("  Computing BERTScore (this may take a while on CPU) ...")
    # BERTScore requires loading a BERT model → slow on first run
 
    P, R, F1 = bert_score_fn(
        cands=generated_texts,
        # cands: candidate (generated) texts → what we're evaluating
 
        refs=reference_texts,
        # refs: reference (ground truth) texts → what we compare against
 
        lang="en",
        # uses English BERT model for embeddings
 
        rescale_with_baseline=False,
        # False: returns raw cosine similarity (0.85-0.95 range)
        # True: rescales against Common Crawl baseline → lower scores
        #       (problematic for simple children's text domain)
 
        device=device,
        # run BERT on GPU if available → much faster
 
        verbose=False,
        # suppress progress bar
    )
    # P, R, F1: tensors of per-story scores, shape [n_stories]
 
    return {
        "BERTScore-P":  P.mean().item(),
        # average precision across all stories
        # .mean(): average over all stories → scalar tensor
        # .item(): converts tensor to Python float
 
        "BERTScore-R":  R.mean().item(),
        # average recall
 
        "BERTScore-F1": F1.mean().item(),
        # average F1 (most commonly reported)
    }
 
 
def compute_mauve(generated_texts, reference_texts, device, max_text_length=256):
    """
    Computes MAUVE score.
 
    MAUVE measures distributional similarity between generated and reference text
    by comparing their probability distributions in GPT-2 embedding space.
 
    Score range: 0 to 1
    Higher = generated distribution closer to real data distribution.
    0.83+ is considered strong for story generation.
    """
 
    try:
        import mauve as mauve_lib
    except ImportError:
        raise ImportError(
            "mauve-text is not installed. Run:  pip install mauve-text"
        )
    # imports mauve inside function → graceful fallback if not installed
 
    print("  Computing MAUVE (featurising texts with GPT-2, may take a while) ...")
    # MAUVE runs GPT-2 to extract features → slow on first run
 
    device_id = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
    # device_id=0: use GPU 0
    # device_id=-1: use CPU
    # MAUVE uses its own device handling (not standard PyTorch)
 
    out = mauve_lib.compute_mauve(
        p_text=reference_texts,
        # p = reference distribution (real human-written stories)
        # "p" = true distribution in MAUVE paper notation
 
        q_text=generated_texts,
        # q = model distribution (generated stories)
        # "q" = approximate distribution in MAUVE paper notation
 
        device_id=device_id,
        # which device to run GPT-2 feature extraction on
 
        max_text_length=max_text_length,
        # truncate texts to 256 tokens for GPT-2 featurisation
        # keeps computation manageable
 
        verbose=False,
        # suppress MAUVE progress output
 
        featurize_model_name="gpt2",
        # use GPT-2 (smallest, 117M) to extract text features
        # swap for "gpt2-large" for better quality if GPU available
    )
 
    return {"MAUVE": out.mauve}
    # out.mauve: scalar MAUVE score between 0 and 1
 
 
def compute_distinct(generated_texts):
    """
    Computes Distinct-1 and Distinct-2 across all generated stories combined.
 
    Distinct-n = unique n-grams / total n-grams
    Measures lexical diversity at corpus level.
    Higher = more diverse vocabulary used across stories.
    """
 
    all_words   = []   # all unigrams from all stories combined
    all_bigrams = []   # all bigrams from all stories combined
 
    for text in generated_texts:
        words = tokenize_words(text)
        # tokenise story into word list
 
        all_words.extend(words)
        # add all words from this story to combined list
 
        all_bigrams.extend(zip(words[:-1], words[1:]))
        # create bigrams by pairing consecutive words
        # zip(["a","b","c"], ["b","c"]) → [("a","b"), ("b","c")]
        # words[:-1]: all words except last
        # words[1:]:  all words except first
 
    d1 = len(set(all_words)) / max(len(all_words), 1)
    # Distinct-1: unique unigrams / total unigrams
    # set(all_words): removes duplicates → count of unique words
    # max(..., 1): prevents division by zero
 
    d2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    # Distinct-2: unique bigrams / total bigrams
    # set of tuples: each unique (word1, word2) pair counted once
 
    return {"Distinct-1": d1, "Distinct-2": d2}
 
 
def compute_repetition(token_ids_list, n=4):
    """
    Computes average 4-gram repetition rate across all generated stories.
 
    Repetition rate = repeated n-grams / total n-grams (within each story)
    Lower = less repetition (better)
    0.03 = only 3% of 4-grams repeated → very good
    """
 
    rates = []
    # stores repetition rate for each story
 
    for token_ids in token_ids_list:
        # process each story's token IDs separately
 
        if len(token_ids) < n:
            rates.append(0.0)
            continue
            # story too short to have any 4-grams → rate = 0, skip
 
        ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
        # extract all 4-grams from this story
        # tuple(token_ids[i:i+4]) for each position i
        # e.g. ids=[1,2,3,4,5] → [(1,2,3,4), (2,3,4,5)]
 
        counts = Counter(ngrams)
        # counts occurrences of each unique 4-gram
        # e.g. {(1,2,3,4): 3, (2,3,4,5): 1}
 
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        # counts EXTRA occurrences (beyond the first) of each repeated 4-gram
        # count=3 → contributes 2 repeated instances
        # count=1 → not repeated, contributes 0
 
        rates.append(repeated / len(ngrams))
        # repetition rate for this story = repeated / total 4-grams
 
    return np.mean(rates)
    # average repetition rate across all N stories
 
 
def compute_vocab_coverage(generated_texts, tokenizer):
    """
    Computes fraction of the full BPE vocabulary used across all generated stories.
    Higher = model uses more of its vocabulary (more diverse).
    6.87% = uses ~2,198 of 32,000 tokens (expected for simple children's stories)
    """
 
    vocab_size  = tokenizer.get_vocab_size()
    # total vocabulary size = 32,000
 
    used_tokens = set()
    # set of unique token IDs seen across all generated stories
 
    for text in generated_texts:
        enc = tokenizer.encode(text)
        # tokenise story into token IDs
 
        used_tokens.update(enc.ids)
        # add all token IDs from this story to the set
        # set.update: adds multiple items, duplicates ignored
 
    return len(used_tokens) / vocab_size
    # fraction of vocabulary used
    # e.g. 2,198 unique tokens / 32,000 = 0.0687 = 6.87%
 
 
def avg_story_length(token_ids_list):
    """Returns average number of BPE tokens per generated story."""
    return np.mean([len(t) for t in token_ids_list])
    # len(t): number of tokens in each story
    # np.mean: average across all stories
    # e.g. [180, 175, 190, ...] → 180.7
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main(args):
 
    os.makedirs("logs", exist_ok=True)
    # creates logs/ directory for saving results
    # exist_ok=True: no error if already exists
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use GPU if available → much faster for generation and BERTScore
 
    # print evaluation configuration
    print("\n" + "=" * 60)
    print("  Evaluation Script")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  N stories  : {args.n_stories}")
    print("=" * 60 + "\n")
 
    # ── Load model + tokeniser ────────────────────────────────────────────────
 
    model, cfg = load_model(args.ckpt, device)
    # loads GPT model from checkpoint, sets to eval mode
 
    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    # loads BPE tokenizer from path stored in model config
    # cfg.tokenizer_path = "tokenizer/tokenizer.json"
    # same tokenizer used during training → consistent vocabulary
 
    print(f"  Tokeniser  : {cfg.tokenizer_path}  (vocab={tokenizer.get_vocab_size():,})\n")
 
    # ── Load real TinyStories as reference ────────────────────────────────────
 
    print("  Loading TinyStories references ...")
    ds = load_dataset("roneneldan/TinyStories", split="train",
                      trust_remote_code=True)
    # loads full TinyStories training split from HuggingFace
    # trust_remote_code=True: required for some HuggingFace datasets
 
    start_idx = int(0.75 * len(ds))
    # start of held-out 25% (model never trained on these)
    # 75% used for training → 25% held out for evaluation
    # ensures fair comparison (no data leakage)
 
    ref_pool = ds.select(range(start_idx, min(start_idx + args.n_stories * 2, len(ds))))
    # selects 2× n_stories stories from held-out 25%
    # 2× buffer in case some stories need to be filtered
    # min(..., len(ds)): ensures we don't go beyond dataset end
 
    ref_texts = ref_pool["text"][:args.n_stories]
    # takes exactly n_stories reference texts
    # ["text"]: extracts the text field from each story
 
    print(f"  References : {len(ref_texts)} stories from held-out 25%\n")
 
    # ── Extract prompts from reference stories ────────────────────────────────
 
    def extract_prompt(text):
        words = text.strip().split()[:6]
        return " ".join(words)
        # takes first 6 words of reference story as prompt
        # e.g. "Once upon a time there was a girl..." → "Once upon a time there was"
        # ensures generated story starts from same context as reference
 
    prompts = [extract_prompt(r) for r in ref_texts]
    # creates one prompt per reference story
 
    # ── Generate N stories ────────────────────────────────────────────────────
 
    print(f"  Generating {args.n_stories} stories ...")
    generated_texts   = []   # stores decoded story strings
    generated_tok_ids = []   # stores token ID lists (for repetition/vocab metrics)
 
    for i, prompt in enumerate(prompts):
        text, tok_ids = generate_story(
            model, tokenizer, prompt, cfg, device,
            max_tokens=args.max_tokens,       # default 200 new tokens
            temperature=args.temperature,      # default 0.8
            top_p=args.top_p,                  # default 0.9
        )
        generated_texts.append(text)
        generated_tok_ids.append(tok_ids)
 
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{args.n_stories} stories ...", end="\r")
            # \r: carriage return → overwrites same line (progress display)
 
    print(f"  Generated {args.n_stories}/{args.n_stories} stories ✓        \n")
    # spaces at end to overwrite previous \r line
 
    # ── Print one sample story ────────────────────────────────────────────────
 
    print("  Sample generated story:")
    print("  " + "─" * 50)
    print(f"  Prompt : {prompts[0]}")
    print(f"  Story  : {generated_texts[0][:300]}...")
    # [:300]: show first 300 chars only to keep output readable
    print("  " + "─" * 50 + "\n")
 
    # ── Compute all metrics ───────────────────────────────────────────────────
 
    print("  Computing metrics ...")
 
    val_loss = torch.load(args.ckpt, map_location="cpu").get("val_loss", float("nan"))
    # reload checkpoint to get val_loss
    # map_location="cpu": load on CPU (don't need GPU for just reading metadata)
 
    ppl = math.exp(val_loss)
    # perplexity = e^val_loss
    # e.g. val_loss=1.4830 → ppl=4.41
 
    bleu_scores  = compute_bleu(generated_texts, ref_texts)
    # BLEU-1,2,3,4: n-gram precision
 
    rouge_scores = compute_rouge(generated_texts, ref_texts)
    # ROUGE-1,2,L: n-gram recall + LCS
 
    distinct     = compute_distinct(generated_texts)
    # Distinct-1,2: lexical diversity
 
    rep_rate     = compute_repetition(generated_tok_ids, n=4)
    # 4-gram repetition rate
 
    avg_len      = avg_story_length(generated_tok_ids)
    # average story length in BPE tokens
 
    vocab_cov    = compute_vocab_coverage(generated_texts, tokenizer)
    # fraction of 32K vocab used
 
    # ── BERTScore (optional — skip if library missing) ────────────────────────
 
    if args.skip_bertscore:
        # user passed --skip_bertscore flag → skip entirely
        bert_scores = {"BERTScore-P": float("nan"),
                       "BERTScore-R": float("nan"),
                       "BERTScore-F1": float("nan")}
        print("  BERTScore : skipped (--skip_bertscore flag set)")
    else:
        try:
            bert_scores = compute_bertscore(generated_texts, ref_texts, device)
            # may fail if bert-score not installed
        except ImportError as e:
            bert_scores = {"BERTScore-P": float("nan"),
                           "BERTScore-R": float("nan"),
                           "BERTScore-F1": float("nan")}
            print(f"  BERTScore : skipped ({e})")
            # graceful fallback: NaN values instead of crash
 
    # ── MAUVE (optional — skip if library missing) ────────────────────────────
 
    if args.skip_mauve:
        # user passed --skip_mauve flag → skip entirely
        mauve_scores = {"MAUVE": float("nan")}
        print("  MAUVE     : skipped (--skip_mauve flag set)")
    else:
        try:
            mauve_scores = compute_mauve(
                generated_texts, ref_texts, device,
                max_text_length=args.mauve_max_len,
                # default 256 tokens for GPT-2 featurisation
            )
        except ImportError as e:
            mauve_scores = {"MAUVE": float("nan")}
            print(f"  MAUVE     : skipped ({e})")
            # graceful fallback
 
    # ── Print evaluation report ───────────────────────────────────────────────
 
    print("\n" + "=" * 55)
    print(f"  Evaluation Report  ({args.n_stories} stories)")
    print("=" * 55)
 
    print(f"  {'Perplexity (PPL)':<28} {ppl:>10.4f}")
    # :<28: left-align label in 28 chars
    # :>10.4f: right-align value in 10 chars with 4 decimal places
 
    print("  " + "─" * 40)
    print(f"  {'BLEU-1':<28} {bleu_scores['BLEU-1']:>10.4f}")
    print(f"  {'BLEU-2':<28} {bleu_scores['BLEU-2']:>10.4f}")
    print(f"  {'BLEU-3':<28} {bleu_scores['BLEU-3']:>10.4f}")
    print(f"  {'BLEU-4':<28} {bleu_scores['BLEU-4']:>10.4f}")
    print("  " + "─" * 40)
    print(f"  {'ROUGE-1':<28} {rouge_scores['ROUGE-1']:>10.4f}")
    print(f"  {'ROUGE-2':<28} {rouge_scores['ROUGE-2']:>10.4f}")
    print(f"  {'ROUGE-L':<28} {rouge_scores['ROUGE-L']:>10.4f}")
    print("  " + "─" * 40)
 
    # BERTScore block with NaN handling
    bs_p  = bert_scores["BERTScore-P"]
    bs_r  = bert_scores["BERTScore-R"]
    bs_f1 = bert_scores["BERTScore-F1"]
 
    fmt = lambda v: f"{v:>10.4f}" if not math.isnan(v) else f"{'N/A':>10}"
    # lambda function for formatting:
    # if value is a real number → format with 4 decimal places
    # if value is NaN (skipped) → show "N/A" right-aligned
 
    print(f"  {'BERTScore-Precision':<28} {fmt(bs_p)}")
    print(f"  {'BERTScore-Recall':<28} {fmt(bs_r)}")
    print(f"  {'BERTScore-F1':<28} {fmt(bs_f1)}")
    print("  " + "─" * 40)
 
    # MAUVE block with NaN handling
    mv = mauve_scores["MAUVE"]
    print(f"  {'MAUVE':<28} {fmt(mv)}")
    print("  " + "─" * 40)
 
    print(f"  {'Distinct-1':<28} {distinct['Distinct-1']:>10.4f}")
    print(f"  {'Distinct-2':<28} {distinct['Distinct-2']:>10.4f}")
    print("  " + "─" * 40)
    print(f"  {'Repetition rate (4-gram)':<28} {rep_rate:>10.4f}")
    print(f"  {'Avg story length (tokens)':<28} {avg_len:>10.1f}")
    # :10.1f: 1 decimal place for average length
    print(f"  {'Vocabulary coverage':<28} {vocab_cov*100:>9.2f}%")
    # *100: convert fraction to percentage
    print("=" * 55)
 
    # ── Save results to CSV ───────────────────────────────────────────────────
 
    csv_path = "logs/eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        # newline="": prevents extra blank lines on Windows
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        # header row
 
        writer.writerow(["Perplexity",              round(ppl, 4)])
        writer.writerow(["BLEU-1",                  round(bleu_scores["BLEU-1"], 4)])
        writer.writerow(["BLEU-2",                  round(bleu_scores["BLEU-2"], 4)])
        writer.writerow(["BLEU-3",                  round(bleu_scores["BLEU-3"], 4)])
        writer.writerow(["BLEU-4",                  round(bleu_scores["BLEU-4"], 4)])
        writer.writerow(["ROUGE-1",                 round(rouge_scores["ROUGE-1"], 4)])
        writer.writerow(["ROUGE-2",                 round(rouge_scores["ROUGE-2"], 4)])
        writer.writerow(["ROUGE-L",                 round(rouge_scores["ROUGE-L"], 4)])
 
        writer.writerow(["BERTScore-P",  round(bs_p,  4) if not math.isnan(bs_p)  else "N/A"])
        writer.writerow(["BERTScore-R",  round(bs_r,  4) if not math.isnan(bs_r)  else "N/A"])
        writer.writerow(["BERTScore-F1", round(bs_f1, 4) if not math.isnan(bs_f1) else "N/A"])
        # conditional: save actual value or "N/A" if metric was skipped
 
        writer.writerow(["MAUVE", round(mv, 4) if not math.isnan(mv) else "N/A"])
 
        writer.writerow(["Distinct-1",              round(distinct["Distinct-1"], 4)])
        writer.writerow(["Distinct-2",              round(distinct["Distinct-2"], 4)])
        writer.writerow(["Repetition_rate_4gram",   round(float(rep_rate), 4)])
        # float(rep_rate): convert numpy float64 to Python float for csv
        writer.writerow(["Avg_story_length_tokens", round(float(avg_len), 1)])
        writer.writerow(["Vocab_coverage_pct",      round(vocab_cov * 100, 2)])
        # *100: save as percentage value
 
    print(f"\n  Results saved → {csv_path}")
 
    # ── Save all generated stories to text file ───────────────────────────────
 
    stories_path = "logs/generated_stories.txt"
    with open(stories_path, "w") as f:
        for i, (prompt, story) in enumerate(zip(prompts, generated_texts)):
            f.write(f"=== Story {i+1} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"{story}\n\n")
            # each story separated by blank line for readability
 
    print(f"  Stories saved  → {stories_path}")
    print()
    # saved for qualitative analysis and report appendix
 
 
if __name__ == "__main__":
 
    p = argparse.ArgumentParser()
    # creates argument parser for command line interface
 
    p.add_argument("--ckpt",       required=True,
                   help="Path to checkpoint (.pt file)")
    # required: must always provide checkpoint path
 
    p.add_argument("--n_stories",  type=int,   default=100,
                   help="Number of stories to generate and evaluate")
    # default 100; use 500 for more reliable metrics
 
    p.add_argument("--max_tokens", type=int,   default=200,
                   help="Max new tokens to generate per story")
    # 200 tokens ≈ 150 words → full children's story
 
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature (lower=safer, higher=creative)")
    # 0.8: slightly creative but coherent
 
    p.add_argument("--top_p",      type=float, default=0.9,
                   help="Nucleus sampling top-p cutoff")
    # 0.9: consider tokens covering 90% probability mass
 
    p.add_argument("--skip_bertscore", action="store_true",
                   help="Skip BERTScore computation")
    # action="store_true": flag (True if present, False if absent)
    # useful when bert-score not installed or want faster evaluation
 
    p.add_argument("--skip_mauve", action="store_true",
                   help="Skip MAUVE computation")
    # MAUVE is the slowest metric → skip for quick evaluation
 
    p.add_argument("--mauve_max_len", type=int, default=256,
                   help="Max token length for MAUVE GPT-2 featuriser")
    # truncation length for GPT-2 feature extraction
    # 256 matches model context length
 
    main(p.parse_args())
    # parses all arguments and passes to main()