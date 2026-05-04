"""
IA Decoder — Native Sentence Composer
======================================
A small transformer decoder trained entirely on IA's own cells.
No borrowed voice, no external corpus.

Architecture:
  - Token embeddings initialised from dictionary DNA (384-dim)
  - Cross-attention to DataG retrieval DNA vectors (what IA means to say)
  - 4 transformer decoder layers
  - Weight-tied output projection

Flow:
  DataG retrieves top-K cells  →  their DNA vectors become the context
  Decoder generates word by word, cross-attending to that meaning
  Output: a novel sentence in IA's voice using IA's vocabulary
"""

import torch
import torch.nn as nn


class IADecoder(nn.Module):

    def __init__(self,
                 vocab_size:  int,
                 dna_dim:     int = 384,
                 model_dim:   int = 384,
                 n_heads:     int = 6,
                 n_layers:    int = 4,
                 ffn_dim:     int = 1024,
                 max_len:     int = 64,
                 dropout:     float = 0.1):
        super().__init__()

        self.dna_dim   = dna_dim
        self.model_dim = model_dim
        self.max_len   = max_len

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_len + 4, model_dim)  # +4: BOS/EOS overhead

        # DNA context projection — in case dims ever differ
        self.ctx_proj = (nn.Linear(dna_dim, model_dim, bias=False)
                         if dna_dim != model_dim else nn.Identity())

        # ── Transformer decoder ───────────────────────────────────────────────
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # pre-LN: more stable on small data
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm     = nn.LayerNorm(model_dim)

        # ── Output ────────────────────────────────────────────────────────────
        # Weight tying: output projection reuses embedding weights
        self.output_proj = nn.Linear(model_dim, vocab_size, bias=False)
        self.output_proj.weight = self.token_emb.weight

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'token_emb' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    # ── Forward ───────────────────────────────────────────────────────────────

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

    def forward(self,
                input_ids:   torch.Tensor,    # (B, T)
                context_dna: torch.Tensor,    # (B, K, dna_dim)
                pad_mask:    torch.Tensor = None  # (B, T) True=ignore
                ) -> torch.Tensor:            # (B, T, vocab_size)

        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        memory = self.ctx_proj(context_dna)           # (B, K, model_dim)
        causal = self._causal_mask(T, device)

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=pad_mask,
        )
        x = self.norm(x)
        return self.output_proj(x)

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self,
                 context_dna,           # (K, dna_dim) np array or tensor
                 tokenizer,             # IATokenizer instance
                 max_new_tokens: int   = 40,
                 temperature:    float = 0.85,
                 top_k:          int   = 50,
                 device:         str   = 'cuda') -> str:
        """
        Generate a sentence conditioned on DataG retrieval DNA context.
        Uses top-k sampling — varied output, not greedy repetition.
        """
        self.eval()

        if not isinstance(context_dna, torch.Tensor):
            context_dna = torch.tensor(context_dna, dtype=torch.float32)
        # Accept (dna_dim,) or (K, dna_dim) — normalise to (1, K, dna_dim)
        if context_dna.dim() == 1:
            context_dna = context_dna.unsqueeze(0)         # (1, dna_dim)
        ctx = context_dna.unsqueeze(0).to(device)          # (1, K, dna_dim)

        tokens = torch.tensor([[tokenizer.bos_id]], device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(tokens, ctx)              # (1, T, vocab)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits,
                                           min(top_k, next_logits.size(-1)))
                next_logits[next_logits < topk_vals[-1]] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)

            if next_token.item() == tokenizer.eos_id:
                break

            tokens = torch.cat([tokens, next_token], dim=1)

        generated = tokens[0, 1:].cpu().tolist()   # strip BOS
        return tokenizer.decode(generated)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.token_emb.num_embeddings,
                'dna_dim':    self.dna_dim,
                'model_dim':  self.model_dim,
                'n_heads':    self.decoder.layers[0].self_attn.num_heads,
                'n_layers':   len(self.decoder.layers),
                'ffn_dim':    self.decoder.layers[0].linear1.out_features,
                'max_len':    self.max_len,
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'IADecoder':
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        cfg = checkpoint['config']
        model = cls(**cfg)
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)
