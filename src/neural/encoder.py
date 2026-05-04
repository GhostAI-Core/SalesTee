"""
Training Tee Encoder — small transformer, text → 128-dim normalized vector.
Trained with InfoNCE contrastive loss on (description, content) pairs.
No external model dependencies after training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


class TrainingTeeEncoder(nn.Module):
    """
    Transformer encoder: token IDs → 128-dim L2-normalized vector.

    Architecture:
      embed(vocab, d_model=128) → pos_enc → 4-layer transformer → mean_pool → L2-norm

    Small by design. Fast inference, no GPU required.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 out_dim: int = 128):
        super().__init__()
        self.d_model   = d_model
        self.out_dim   = out_dim
        self.embed     = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-norm — stabler for small data
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = (nn.Linear(d_model, out_dim, bias=False)
                     if out_dim != d_model else nn.Identity())

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_ids:      (B, L) int tensor
        attention_mask: (B, L) bool tensor, True = PAD (ignored)
        Returns:        (B, out_dim) L2-normalized
        """
        x = self.embed(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask  # True = ignore

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Mean pool over non-padding tokens
        if attention_mask is not None:
            mask = (~attention_mask).float().unsqueeze(-1)  # (B, L, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            x = x.mean(dim=1)

        x = self.proj(x)
        return F.normalize(x, dim=-1)

    # ── Convenience ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_text(self, token_ids: list[int], device: str = 'cpu') -> torch.Tensor:
        """Single text → (out_dim,) normalized vector. No grad."""
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        return self(ids).squeeze(0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
