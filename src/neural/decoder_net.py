"""
Steve Decoder — transformer decoder, latent vector → text.
Conditioned on the encoder output at every layer via cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SteveDecoder(nn.Module):
    """
    Autoregressive transformer decoder conditioned on a latent vector.

    The latent is projected to (1, d_model) and prepended as a 'memory'
    sequence — the decoder cross-attends to it at every layer.

    input:  latent (B, latent_dim) + shifted token IDs (B, T)
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

    def forward(self, latent: torch.Tensor, tgt_ids: torch.Tensor,
                tgt_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        latent:       (B, latent_dim)
        tgt_ids:      (B, T) — shifted-right token IDs
        tgt_pad_mask: (B, T) bool — True = PAD
        Returns:      (B, T, vocab_size) logits
        """
        B, T = tgt_ids.shape

        # Memory: project latent to (B, 1, d_model)
        memory = self.latent_proj(latent).unsqueeze(1)

        # Target embeddings + position
        tgt = self.embed(tgt_ids) * math.sqrt(self.d_model)
        tgt = tgt + self.pos_enc[:, :T]

        # Causal mask
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
                 top_k: int = 40) -> list[int]:
        """
        Greedy / top-k sampling from a single latent vector.
        Returns list of token IDs (without BOS, stops at EOS).
        """
        self.eval()
        device = latent.device
        lat    = latent.unsqueeze(0) if latent.dim() == 1 else latent  # (1, dim)

        generated = [bos_id]
        seen_bigrams: set[tuple] = set()

        for _ in range(max_new):
            ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self(lat, ids)          # (1, T, vocab)
            next_logits = logits[0, -1] / temperature

            # Block bigram repetition — zero out tokens that would form a seen bigram
            if len(generated) >= 1:
                last = generated[-1]
                for blocked in list(seen_bigrams):
                    if blocked[0] == last:
                        next_logits[blocked[1]] = float('-inf')

            # Top-k filter
            if top_k > 0:
                vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = vals[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))

            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

            if next_id == eos_id:
                break

            if len(generated) >= 1:
                seen_bigrams.add((generated[-1], next_id))
            generated.append(next_id)

        return generated[1:]  # strip BOS

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
