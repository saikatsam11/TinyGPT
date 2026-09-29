import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.config import ModelConfig


# ── Causal Self-Attention ─────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 #safety check d_model must be divisible by attention heads
        self.n_heads  = cfg.n_heads #attention heads
        self.d_model  = cfg.d_model #parameter model or token embedding size e.g 448
        self.head_dim = cfg.d_model // cfg.n_heads   #each head dimension e.g 448/7 = 64

        # Fused Q, K, V in one matmul
        self.qkv_proj  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)  # single linear layer that computes Q,K,V all at once
        self.out_proj   = nn.Linear(cfg.d_model, cfg.d_model,     bias=False) # Final projection after concatenating all heads 
        self.attn_drop  = nn.Dropout(cfg.dropout) # dropout on attention weights (after softmax)
        self.resid_drop = nn.Dropout(cfg.dropout) # # dropout on final output (before residual addition outside this block)

        # Causal mask — buffer moves with .to(device)
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len)) # mask matriusx
        self.register_buffer("mask", mask.view(1, 1, cfg.context_len, cfg.context_len))

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimension
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2) # Q, K, V matrices

        # Reshape to multi-head format → [B, n_heads, T, head_dim]
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        scale  = 1.0 / math.sqrt(self.head_dim) # scaling factor (1/sqrt(64))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale # compute attention scores (q k^T)
        
        # apply causal mask: future positions → -inf
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # computing attention scores and applying dropout
        attn   = self.attn_drop(F.softmax(scores, dim=-1))
        
        #mutiplying with value matrix
        out = torch.matmul(attn, v)                              # [B, n_heads, T, head_dim]
        #concatenate all the heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # [B, T, d_model]
        #final project with dropout
        return self.resid_drop(self.out_proj(out))


# ── Transformer Block ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        
        # layerNorm before attention
        self.ln1  = nn.LayerNorm(cfg.d_model)

        # Causal multi-head self-attention module
        self.attn = CausalSelfAttention(cfg)

        # LayerNorm before feed-forward network
        self.ln2  = nn.LayerNorm(cfg.d_model)

        # Feed Forward Network
        self.ffn  = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff,    bias=False),  # expand features (e.g., 448 → 1792)
            nn.GELU(),                                        # GELU activation
            nn.Linear(cfg.d_ff,    cfg.d_model, bias=False),  # project back to original size
            nn.Dropout(cfg.dropout),                          # dropout
        )

    def forward(self, x):
        # ----- Attention Block -----
        # 1. Normalize input
        # 2. Apply causal self-attention
        # 3. Add residual connection (input + attention output)
        x = x + self.attn(self.ln1(x))

        # ----- Feed Forward Block -----
        # 1. Normalize input
        # 2. Apply MLP (token-wise transformation)
        # 3. Add residual connection        
        x = x + self.ffn(self.ln2(x))    # pre-norm + residual
        return x # final output of the Transformer block


# ── Decoder only Transformer Model ─────────────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg   # hyperparameters (context_len, layers,..etc)

        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model),  #token embeddings
            pos_emb = nn.Embedding(cfg.context_len, cfg.d_model), #positional embeddings
            drop    = nn.Dropout(cfg.dropout), # dropout after embeddings
            blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]), # stack of decoder only transformer blocks
            ln_f    = nn.LayerNorm(cfg.d_model), # final layer normalization
        ))

        # Language modeling head: maps hidden states → vocabulary logits
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output layer
        self.lm_head.weight = self.transformer.tok_emb.weight

        # Weight initialisation
        self.apply(self._init_weights)
        # Scale residual projections
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "ffn.2.weight")):
                nn.init.normal_(p, 0.0, 0.02 / (2 * cfg.n_layers) ** 0.5)
        
        # Total weights
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT ready — {n_params:,} parameters ({n_params/1e6:.2f}M)")

    #Weights initliazation function
    def _init_weights(self, m):

        #if linear layer, then initialize with small values
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

        #for embeddings
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

        # for layerNorm 
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, idx, targets=None):
        """
        idx:     [B, T]  token IDs
        targets: [B, T]  next-token labels (optional)
        returns: logits [B, T, V],  loss scalar or None
        """

        B, T = idx.shape   # batch size and sequence length

        # Ensure sequence length does not exceed maximum context length
        assert T <= self.cfg.context_len

        # Create position indices
        pos    = torch.arange(T, device=idx.device).unsqueeze(0)  # [1, T]
        
        # token + pos embeddings + dropout
        x      = self.transformer.drop(
                    self.transformer.tok_emb(idx) +
                    self.transformer.pos_emb(pos)
                 )

        # pass through all Transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        # Final normalization
        x      = self.transformer.ln_f(x)     # [B, T, d_model]
        
        # Project to vocabulary size → logits for each token
        logits = self.lm_head(x)              # [B, T, vocab_size]

        loss = None

        # If targets are provided, compute cross-entropy loss
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), # flatten predictions
                targets.view(-1),                     # flatten targets
                ignore_index=-1,                      # ignore padding tokens
            )
        return logits, loss

    # ── Autoregressive generation ─────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9, eos_id=None):
        """
        idx : [B, T] seed token IDs
        Returns [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):

            # Keep only last context_len tokens
            idx_cond      = idx[:, -self.cfg.context_len:]  # keep only last context_len 
            
            # forward pass → get logits
            logits, _     = self(idx_cond)

            # Take logits of last token and apply temperature scaling
            logits        = logits[:, -1, :] / temperature   

            # Nucleus (top-p) sampling

            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)

            # convert logits → probabilities
            probs = F.softmax(sorted_logits, dim=-1)
            
            # Compute cumulative probabilities
            cum_probs = torch.cumsum(probs, dim=-1)

            # Mask tokens where cumulative probability exceeds top_p
            mask = cum_probs > top_p

            # Ensure at least one token is kept
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            # Set filtered tokens to -inf → probability becomes 0 after softmax
            sorted_logits[mask] = float("-inf")

            # Scatter back to original ordering
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            # Append token
            idx = torch.cat([idx, next_tok], dim=1)

            # Stop if EOS reached for all batches
            if eos_id is not None and (next_tok == eos_id).all():
                break

        return idx

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def configure_optimizer(self, cfg: ModelConfig):
        """AdamW with weight decay on matrices only, not biases / LN params."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            (decay if p.dim() >= 2 else no_decay).append(p)

        groups = [
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        print(f"AdamW | decay={sum(p.numel() for p in decay):,} "
              f"| no_decay={sum(p.numel() for p in no_decay):,}")
        return optimizer
