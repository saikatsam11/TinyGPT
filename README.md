# InstructStoryLM — Lightweight Instruction-Tuned Story Generator

A compact 31.3M parameter GPT-style language model trained from 
scratch for children's story generation, with instruction fine-tuning 
support and comprehensive evaluation. Trained entirely on a single 
NVIDIA RTX 4060 Ti GPU in under 3 hours.

---

## Model Overview

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Parameters | 31.3M |
| Layers | 7 |
| Hidden size | 448 |
| Attention heads | 7 |
| Context length | 256 (pretraining) / 512 (fine-tuning) |
| Vocabulary size | 32,000 (BPE) |
| Pretraining data | TinyStories (75% subset, ~330M tokens) |
| Fine-tuning data | 3,075 instruction-response pairs |
| Hardware | NVIDIA RTX 4060 Ti 16GB |
| Training time | ~1.71 hours (pretraining) |

---

## Results

| Metric | Value |
|---|---|
| Perplexity (PPL) | 4.35 |
| BLEU-1 | 0.336 |
| ROUGE-1 | 0.381 |
| BERTScore F1 | 0.869 |
| MAUVE | 0.835 |
| Repetition Rate (4-gram) | 0.033 |

---

## Project Structure

```
InstructStoryLM/
├── model/
│   ├── gpt.py                # GPT model architecture
│   └── config.py             # ModelConfig dataclass
├── tokenizer/
│   └── tokenizer.json        # Trained BPE tokenizer
├── data/
│   └── setup_data.py         # Download + tokenize + shard dataset
├── train.py                  # Pretraining script
├── final_data_cleaned.jsonl  # finetune dataset
├── finetune.py               # Instruction fine-tuning script
├── generate.py               # Story generation (pretrained model)
├── finetune_inference.py     # Instruction-based generation
├── evaluate.py               # Full evaluation pipeline
├── requirements.txt
└── README.md
```

---

## Pretrained & Fine-tuned Checkpoints

All model checkpoints are hosted on HuggingFace:  
👉 [saix11/InstructStoryLM](https://huggingface.co/saix11/InstructStoryLM/tree/main)

| Checkpoint | Description |
|---|---|
| `ckpt_best.pt` | Best pretrained checkpoint (val PPL = 4.41) |
| `ft_ckpt_best.pt` | Best instruction fine-tuned checkpoint |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download checkpoints
```bash
# Checkpoints are downloaded automatically via the code below
# or manually from HuggingFace
```

---

## Usage

### Pretraining

```bash
# Step 1 — Download and tokenize dataset (run once)
python data/setup_data.py

# Step 2 — Train
python train.py
```

### Instruction Fine-tuning

```bash
python finetune.py
```
- Trains on instruction-response pairs in `.jsonl` format  
- Applies response-only loss masking  
- Saves checkpoints to `checkpoints_ft/`

### Story Generation (Pretrained Model)

```bash
# One-shot
python generate.py --ckpt ckpt_best.pt \
                   --prompt "Once upon a time there was a little girl"

# Interactive mode
python generate.py --ckpt ckpt_best.pt

# With custom sampling settings
python generate.py --ckpt ckpt_best.pt \
                   --prompt "Once upon a time" \
                   --temperature 0.8 \
                   --top_p 0.9 \
                   --max_tokens 300
```

### Instruction-Based Generation (Fine-tuned Model)

```bash
python finetune_inference.py
```

### Evaluation

```bash
python evaluate.py --ckpt ckpt_best.pt --n_stories 500
```

Metrics computed: PPL, BLEU-1/2/3/4, ROUGE-1/2/L, BERTScore, 
MAUVE, Distinct-1/2, Repetition Rate, Vocabulary Coverage

Outputs saved to:
```
logs/eval_results.csv
logs/generated_stories.txt
```

---

## Load Model Programmatically

```python
from huggingface_hub import hf_hub_download
import torch
from model.gpt import GPT
from model.config import ModelConfig

# Download checkpoint
path = hf_hub_download(
    repo_id="saix11/InstructStoryLM",
    filename="ckpt_best.pt"
)

# Load model
cfg   = ModelConfig()
model = GPT(cfg)
model.load_state_dict(torch.load(path, map_location="cpu"))
model.eval()
```

---

## Example Output
### Pretrained Model — Story Generation

**Prompt:** `Once upon a time there was a cat`

**Generated:**
> Once upon a time there was a cat called Sam. Sam was a cool cat, always looking for ways to make friends.
> One day, Sam had a great idea. He wanted to bake something special, so he went to the store to find some ngredients. 
> Sam looked at all the ingredients he needed, but he couldn't find any that he could use.
> He asked the store clerk for help....

### Instruction-Tuned Model

**Instruction:** `Write a short bedtime story about a rabbit who learns patience`

> Once upon a time, in a sunny meadow, there lived a little 
> rabbit named Rosie. Rosie loved to play with her friends, 
> but she often gave up when things got tough. One day, while 
> hopping through the meadow, Rosie stumbled upon a shiny...

---

**Instruction:** `Rewrite "A little bird sang a cheerful song" in a funny way`

> In a bright green forest, a little bird named Zephyr perched 
> on a branch, watching the world go by. One day, while she was 
> singing, she met a friendly rabbit named Rosie. Rosie was lost 
> and didn't know how to get back to her burrow. Zephyr's...

---

## Notes

- Model checkpoints are not stored in this repository due to 
  size constraints — download from HuggingFace
- The tokenizer (`tokenizer/tokenizer.json`) must match the 
  checkpoint used for inference
- All training was performed without any pretrained weights — 
  the model is trained from scratch

---