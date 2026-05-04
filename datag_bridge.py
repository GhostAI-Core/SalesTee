"""
DataG Bridge — Steve's Memory Backend
=======================================
Connects Steve's neuronal architecture to the substrate.

Encoding:  SteveEncoder (128-dim, trained on Steve's own cells)
Decoding:  SteveDecoder (generates text from latent vector)
Storage:   LivingCell methodology cells in data_store/methodology/
Retrieval: Cosine similarity field walk

Cell source_table labels:
  meth_identity     — who Steve is
  meth_code         — code patterns Steve knows
  meth_reasoning    — how Steve thinks
  meth_conversation — learned during live conversation (future)

Used by:
  n1  — concept lookup (get_node)
  n15 — association (field walk)
  n17 — generation (field reader + decoder)
  main.py — conversational shortcut
"""

import os
import sys
import json
import time
import numpy as np

_STEVE_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATAG_SRC  = os.path.abspath(os.path.join(_STEVE_ROOT, '..', 'DataG', 'src'))
_DATAG_ROOT = os.path.abspath(os.path.join(_STEVE_ROOT, '..', 'DataG'))
_MODEL_DIR  = os.path.join(_STEVE_ROOT, 'models')
_SRC_DIR    = os.path.join(_STEVE_ROOT, 'src')

for p in (_DATAG_SRC, _DATAG_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


class DataGBridge:
    """
    Singleton bridge to Steve's substrate.
    Loaded once at startup, shared across all neurons.

    Encoding is fully handled by SteveEncoder — no external model dependency.
    """
    _instance = None

    @classmethod
    def get(cls) -> 'DataGBridge':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._ready    = False
        self.substrate = None
        self._enc      = None   # SteveEncoder
        self._dec      = None   # SteveDecoder
        self._tok      = None   # SteveTokenizer
        self._device   = 'cpu'
        self._load()

    def _load(self):
        try:
            self._load_substrate()
            self._load_models()
            self._ready = True
        except Exception as e:
            print(f"[Bridge] Load failed: {e}")
            import traceback; traceback.print_exc()

    def _load_substrate(self):
        from system import AgenticSystem
        print("[Bridge] Loading substrate…", flush=True)
        self.substrate = AgenticSystem(hdc_dim=384, slim=True, skip_cells=True)
        print(f"[Bridge] {len(self.substrate.methodology_cells):,} cells online", flush=True)

    def _load_models(self):
        import torch
        from tokenizer import SteveTokenizer
        from encoder import SteveEncoder
        from decoder_net import SteveDecoder

        tok_path = os.path.join(_MODEL_DIR, 'tokenizer.json')
        enc_path = os.path.join(_MODEL_DIR, 'encoder.pt')
        dec_path = os.path.join(_MODEL_DIR, 'decoder.pt')

        if not all(os.path.exists(p) for p in (tok_path, enc_path, dec_path)):
            print("[Bridge] Models not found — run train_steve.py first")
            return

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tok    = SteveTokenizer.load(tok_path)

        self._enc = SteveEncoder(
            vocab_size=self._tok.vocab_size, d_model=128, out_dim=128,
            nhead=4, num_layers=4, dim_feedforward=256,
        )
        self._enc.load_state_dict(torch.load(enc_path, map_location=self._device))
        self._enc.to(self._device).eval()

        self._dec = SteveDecoder(
            vocab_size=self._tok.vocab_size, latent_dim=128, d_model=128,
            nhead=4, num_layers=4, dim_feedforward=256,
        )
        self._dec.load_state_dict(torch.load(dec_path, map_location=self._device))
        self._dec.to(self._device).eval()

        print(f"[Bridge] SteveEncoder + SteveDecoder loaded ({self._device})")

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Text → 128-dim L2-normalised vector via SteveEncoder."""
        if self._enc is None:
            raise RuntimeError("Encoder not loaded")
        import torch
        ids  = self._tok.encode(text, max_len=128)
        tens = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            vec = self._enc(tens).squeeze(0).cpu().numpy()
        return vec.astype(np.float32)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def top_cells(self, query: str, k: int = 10) -> list:
        """
        Return top-K (sim, cell_id, cell) tuples most similar to query.
        Used by n15 for association.
        """
        if not self._ready:
            return []
        q_vec   = self.encode(query)
        results = []
        for cid, cell in self.substrate.methodology_cells.items():
            dna = getattr(cell, 'dna', None)
            if dna is None or len(dna) == 0:
                continue
            dv  = np.array(dna, dtype=np.float32)
            dn  = np.linalg.norm(dv)
            if dn < 1e-8:
                continue
            sim = float(np.dot(q_vec, dv / dn))
            results.append((sim, cid, cell))
        results.sort(reverse=True)
        return results[:k]

    def field_read(self, query: str, k: int = 5) -> str | None:
        """
        Find closest cell, try decoder generation first, fall back to raw content.

        source_table routing:
          meth_code      → decoder generates code from latent
          meth_identity  → decoder generates prose from latent
          meth_reasoning → decoder generates reasoning from latent
        All types: if decoder output is too short/empty, return raw cell content.
        """
        if not self._ready:
            return None
        hits = self.top_cells(query, k=k)
        if not hits:
            return None

        best_sim, best_cid, best_cell = hits[0]
        if best_sim < 0.20:
            return None

        # Code cells: return raw content — decoder can't reproduce exact syntax
        source = getattr(best_cell, 'source_table', '')
        if 'code' in source:
            content = (getattr(best_cell, 'content', '') or '').strip()
            return content if len(content) > 5 else None

        # Prose cells: try decoder generation first
        if self._dec is not None and self._enc is not None:
            generated = self._generate(query)
            if generated and len(generated.strip()) > 8:
                return generated

        # Fall back to raw cell content
        content = (getattr(best_cell, 'content', '') or '').strip()
        return content if len(content) > 5 else None

    def _generate(self, query: str, temperature: float = 0.75,
                  top_k: int = 40, max_new: int = 80) -> str | None:
        """Encode query → run decoder → decode output tokens → text."""
        try:
            import torch
            ids   = self._tok.encode(query, max_len=128)
            tens  = torch.tensor([ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                latent = self._enc(tens).squeeze(0)
                gen_ids = self._dec.generate(
                    latent,
                    bos_id=self._tok.bos_id,
                    eos_id=self._tok.eos_id,
                    max_new=max_new,
                    temperature=temperature,
                    top_k=top_k,
                )
            return self._tok.decode(gen_ids, skip_special=True)
        except Exception as e:
            return None

    def get_concept(self, word: str) -> dict | None:
        """
        Lookup a concept by name. Returns the best matching cell's metadata.
        Used by n1 as a drop-in for SQLite concept lookup.
        """
        if not self._ready:
            return None
        hits = self.top_cells(word, k=1)
        if not hits:
            return None
        sim, cid, cell = hits[0]
        if sim < 0.30:
            return None
        return {
            'word':    word,
            'cell_id': cid,
            'content': getattr(cell, 'content', ''),
            'sim':     sim,
            'dna':     getattr(cell, 'dna', []),
            'cell':    cell,
        }

    # ── Cell writing ─────────────────────────────────────────────────────────

    def save_cell(self, description: str, content: str,
                  source_table: str = 'conversation',
                  confidence: float = 0.85, energy: float = 120.0) -> str:
        """
        Encode + persist a new cell to Steve's substrate.
        Returns the cell_id.

        source_table should be one of:
          'identity', 'code', 'reasoning', 'conversation'
        The 'meth_' prefix is added automatically.
        """
        import uuid, json as _json
        import numpy.random as npr

        cid = f"meth_{source_table}_{uuid.uuid4().hex[:12]}"
        dna = self.encode(description).tolist()

        dim, rank = 128, 4
        cell = {
            "id":               cid,
            "content":          content,
            "source_table":     f"meth_{source_table}",
            "confidence":       confidence,
            "source":           "conversation",
            "dna":              dna,
            "W_down":           (npr.randn(dim, rank) * 0.01).tolist(),
            "W_up":             (npr.randn(rank, dim) * 0.01).tolist(),
            "connections":      {},
            "energy":           energy,
            "activation_count": 0,
            "last_activated":   0.0,
            "birth_time":       time.time(),
            "meta":             {"description": description},
        }
        path = os.path.join(_STEVE_ROOT, 'data_store', 'methodology', f"{cid}.json")
        with open(path, 'w') as f:
            _json.dump(cell, f)

        # Hot-load into live substrate
        if self.substrate is not None:
            from living_cell import LivingCell
            lc = LivingCell.from_dict(cell)
            self.substrate.methodology_cells[cid] = lc

        return cid

    @property
    def ready(self) -> bool:
        return self._ready
