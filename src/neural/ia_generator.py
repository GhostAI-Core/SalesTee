"""
IA Generator — Decoder Inference Interface
===========================================
Loads the trained decoder once at startup.
Called after DataG field walk to compose a novel sentence conditioned
on the conjugate attractor vector and the walked cell DNA vectors.

Context layout (cross-attention slots):
  slot 0 — conjugate vector (what IA intends to say)
  slot 1..K — walked cell DNA vectors (supporting context)

Usage:
    generator = IAGenerator()
    text = generator.compose(walk_cells, conj_vec)
"""

import os
import numpy as np

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_DATAG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '../../'))
_MODELS_DIR = os.path.join(_DATAG_ROOT, 'models')

TOKENIZER_PATH = os.path.join(_MODELS_DIR, 'ia_tokenizer.json')
DECODER_PATH   = os.path.join(_MODELS_DIR, 'ia_decoder.pt')

DNA_DIM = 384


class IAGenerator:
    """
    Wraps the trained decoder for inference.
    Loaded once — compose() is called per query.
    Falls back silently if model files are not found.
    """

    def __init__(self, device: str = 'cpu', temperature: float = 0.75,
                 top_k: int = 50, max_tokens: int = 40):
        self.device      = device
        self.temperature = temperature
        self.top_k       = top_k
        self.max_tokens  = max_tokens
        self._ready      = False
        self._encoder    = None
        self._load()

    def _load(self):
        if not os.path.exists(TOKENIZER_PATH) or not os.path.exists(DECODER_PATH):
            return
        try:
            from neural.ia_tokenizer import IATokenizer
            from neural.ia_decoder   import IADecoder

            self.tokenizer = IATokenizer.load(TOKENIZER_PATH)
            self.model     = IADecoder.load(DECODER_PATH, device=self.device)
            self.model.eval()
            self._ready    = True
        except Exception as e:
            print(f"[IAGenerator] Could not load decoder: {e}")

        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self._encoder = None

    @property
    def ready(self) -> bool:
        return self._ready

    def compose(self, cells: list, conj_vec=None, n_ctx: int = 3) -> str | None:
        """
        Generate a novel sentence conditioned on the conjugate attractor
        and the DNA of the top walked cells.

        conj_vec — (384,) numpy array from FieldReader._conjugate_vec()
        cells    — list of LivingCell from the field walk
        n_ctx    — number of cell DNA slots (conj_vec occupies slot 0)
        """
        if not self._ready:
            return None

        import torch

        context_vecs = []

        # Slot 0: conjugate vector — primary semantic intent
        if conj_vec is not None:
            cv = np.array(conj_vec, dtype=np.float32)
            cn = np.linalg.norm(cv)
            if cn > 1e-8:
                context_vecs.append(cv / cn)

        # Slots 1..n_ctx: walked cell DNA vectors
        for cell in cells[:n_ctx]:
            dna = getattr(cell, 'dna', None)
            if dna is None:
                continue
            v = np.array(dna, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn > 1e-8:
                context_vecs.append(v / vn)

        if not context_vecs:
            return None

        context = np.stack(context_vecs, axis=0)  # (K, 384)
        conj_unit = context_vecs[0] if context_vecs else None

        try:
            result = self.model.generate(
                context,
                self.tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                device=self.device,
            )
            if not result or len(result.strip()) <= 8:
                return None
            # Quality gate: discard if output isn't semantically on-topic
            if conj_unit is not None and self._encoder is not None:
                try:
                    out_vec = self._encoder.encode(result, convert_to_numpy=True)
                    out_vec = out_vec / (np.linalg.norm(out_vec) + 1e-8)
                    sim = float(np.dot(conj_unit, out_vec))
                    if sim < 0.20:
                        return None
                except Exception:
                    pass
            return result
        except Exception:
            return None
