import numpy as np
import typing as t
import os

# ── Custom Encoder Foundation ─────────────────────────────────────────────────
# Uses Training Tee's own encoder + tokenizer. Zero external model dependencies.

import sys
import torch

_NEURAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'neural')
sys.path.insert(0, _NEURAL_DIR)

from tokenizer import TrainingTeeTokenizer
from encoder import TrainingTeeEncoder

_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))


class HDCSubstrate:
    """Provides real semantic encoding using Training Tee's custom encoder."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.model = None
        self.tokenizer = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tok_path = os.path.join(_MODEL_DIR, 'tokenizer.json')
        enc_path = os.path.join(_MODEL_DIR, 'encoder.pt')

        if os.path.exists(tok_path) and os.path.exists(enc_path):
            try:
                self.tokenizer = TrainingTeeTokenizer.load(tok_path)
                self.model = TrainingTeeEncoder(
                    vocab_size=self.tokenizer.vocab_size,
                    d_model=128, out_dim=128, nhead=4,
                    num_layers=4, dim_feedforward=256, dropout=0.0,
                ).to(self._device)
                self.model.load_state_dict(
                    torch.load(enc_path, map_location=self._device, weights_only=True)
                )
                self.model.eval()
                self.dimension = 128
                print(f"[Substrate] Custom encoder ready (128-dim, vocab={self.tokenizer.vocab_size})")
            except Exception as e:
                print(f"[Substrate] Custom encoder load failed ({e}) — using hash fallback")
                self.model = None
                self.tokenizer = None
        else:
            print(f"[Substrate] No trained models found — using hash-based HDC fallback (dim={self.dimension})")

    def encode(self, text: str) -> np.ndarray:
        """Converts raw text into a semantic vector using the custom encoder."""
        if self.model and self.tokenizer:
            ids = self.tokenizer.encode(text, max_len=128)
            t_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            with torch.no_grad():
                vec = self.model(t_ids).squeeze(0).cpu().numpy()
            return vec
        return self._hash_encode(text)

    def _hash_encode(self, text: str) -> np.ndarray:
        """Deterministic hash-based HDC encoding. Same text -> same vector."""
        import hashlib
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)
        for i, word in enumerate(words):
            seed = int(hashlib.md5(word.encode()).hexdigest(), 16) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            word_vec = rng.randn(self.dimension).astype(np.float32)
            vec += word_vec
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dimension).astype(np.float32)
            norm = np.linalg.norm(vec)
        return vec / norm

    def random_hypervector(self) -> np.ndarray:
        """Generates a random unit vector (fallback)."""
        vec = np.random.randn(self.dimension)
        return vec / np.linalg.norm(vec)

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """XOR operation for binding (quasi-orthogonal). Output is bipolar."""
        return v1 * v2

    def bundle(self, vectors: t.List[np.ndarray]) -> np.ndarray:
        """Superposition operation (majority rule). Output is bipolar."""
        summed = np.sum(vectors, axis=0)
        # Handle ties randomly or with bias
        summed[summed == 0] = np.random.choice([-1, 1], size=np.sum(summed == 0))
        return np.sign(summed).astype(int)

    def permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """Cyclic shift operation for preserving order or sequence."""
        return np.roll(v, shift)

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        # Custom encoder outputs L2-normalized vectors
        return float(np.dot(v1, v2))


class AssociativeMemory:
    """A simple associative memory store using HDC."""
    def __init__(self, substrate: HDCSubstrate):
        self.substrate = substrate
        self.entries: t.Dict[str, np.ndarray] = {}

    def store(self, label: str, vector: np.ndarray):
        self.entries[label] = vector

    def search(self, query: np.ndarray, threshold: float = 0.3) -> t.List[t.Tuple[str, float]]:
        """Finds items in memory that match the query vector."""
        results = []
        for label, stored_vec in self.entries.items():
            sim = self.substrate.similarity(query, stored_vec)
            if sim >= threshold:
                results.append((label, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)
