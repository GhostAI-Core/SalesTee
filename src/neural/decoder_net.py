"""
Training Tee Decoder — transformer decoder, latent vector → text.
Conditioned on the encoder output at every layer via cross-attention.

Stage 1 (current, corpus < 1000 cells):
  latent is (B, latent_dim) — single best-match cell DNA.
  Decoder is NOT used in the live response path (retrieval-only).
  Kept trained so Stage 2 upgrade is a flag flip, not a rebuild.

Stage 2 (corpus 1000+ cells, retrained encoder):
  Pass latent (B, K, latent_dim) — top-K cell DNAs ranked by sim.
  Decoder cross-attends to all K memory positions and learns to blend
  top-K retrieved cells into a coherent response.
  Activate by passing a (B, K, latent_dim) tensor to forward/generate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TrainingTeeDecoder(nn.Module):
    """
    Autoregressive transformer decoder conditioned on a latent vector
    or a ranked sequence of latent vectors (top-K cell DNAs).

    Stage 1: latent (B, latent_dim)       → memory (B, 1, d_model)
    Stage 2: latent (B, K, latent_dim)    → memory (B, K, d_model)
             weights (B, K) sim scores    → memory slots scaled by rank

    input:  latent (B, latent_dim) or (B, K, latent_dim)
    output: logits (B, T, vocab_size)
    """

    def __init__(self, vocab_size: int, latent_dim: int = 128,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 256,
                 dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.max_len    = max_len

        self.latent_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.embed       = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc     = self._build_pe(max_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj    = nn.Linear(d_model, vocab_size, bias=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> nn.Parameter:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _build_memory(self, latent: torch.Tensor,
                      weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Stage 1: latent (B, dim)    → (B, 1, d_model)
        Stage 2: latent (B, K, dim) → (B, K, d_model), sim-weighted by rank
        """
        if latent.dim() == 2:
            # Stage 1 — single latent
            return self.latent_proj(latent).unsqueeze(1)   # (B, 1, d_model)

        # Stage 2 — top-K latents
        memory = self.latent_proj(latent)                  # (B, K, d_model)
        if weights is not None:
            # Scale memory slots by softmax of similarity scores
            # so the highest-ranked cell has the strongest signal
            w = torch.softmax(weights, dim=-1).unsqueeze(-1)  # (B, K, 1)
            memory = memory * w
        return memory

    def forward(self, latent: torch.Tensor, tgt_ids: torch.Tensor,
                tgt_pad_mask: torch.Tensor | None = None,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        latent:       (B, latent_dim) Stage 1 | (B, K, latent_dim) Stage 2
        tgt_ids:      (B, T) shifted-right token IDs
        tgt_pad_mask: (B, T) bool, True = PAD
        weights:      (B, K) similarity scores — Stage 2 only
        Returns:      (B, T, vocab_size) logits
        """
        B, T = tgt_ids.shape

        memory = self._build_memory(latent, weights)       # (B, 1|K, d_model)

        # Target embeddings + positional encoding
        tgt = self.embed(tgt_ids) * math.sqrt(self.d_model)
        tgt = tgt + self.pos_enc[:, :T]

        # Causal mask — prevent attending to future tokens
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)

        out = self.transformer(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_is_causal=True,
        )
        return self.out_proj(out)  # (B, T, vocab_size)

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, latent: torch.Tensor, bos_id: int, eos_id: int,
                 max_new: int = 128, temperature: float = 0.8,
                 top_k: int = 40,
                 weights: torch.Tensor | None = None) -> list[int]:
        """
        Autoregressive generation from a single latent (Stage 1) or
        a top-K latent stack (Stage 2).

        latent:  (latent_dim,) or (1, latent_dim) for Stage 1
                 (1, K, latent_dim) for Stage 2
        weights: (1, K) similarity scores — Stage 2 only
        Returns: list of token IDs (no BOS, stops at EOS)
        """
        self.eval()
        device = latent.device

        # Normalise to batch dim
        if latent.dim() == 1:
            lat = latent.unsqueeze(0)          # (1, dim)
        elif latent.dim() == 2 and latent.shape[0] != 1:
            lat = latent.unsqueeze(0)          # (1, K, dim) from (K, dim)
        else:
            lat = latent                       # already (1, dim) or (1, K, dim)

        generated = [bos_id]
        seen_bigrams: set[tuple] = set()

        for _ in range(max_new):
            ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self(lat, ids, weights=weights)   # (1, T, vocab)
            next_logits = logits[0, -1] / temperature

            # Block bigram repetition
            if len(generated) >= 1:
                last = generated[-1]
                for blocked in seen_bigrams:
                    if blocked[0] == last:
                        next_logits[blocked[1]] = float('-inf')

            # Top-k filter
            if top_k > 0:
                vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(
                    next_logits < vals[-1], float('-inf')
                )

            probs   = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

            if next_id == eos_id:
                break

            seen_bigrams.add((generated[-1], next_id))
            generated.append(next_id)

        return generated[1:]  # strip BOS

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
