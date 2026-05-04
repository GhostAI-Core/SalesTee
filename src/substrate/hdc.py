import numpy as np
import typing as t
import os

# Real Neural Foundation
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class HDCSubstrate:
    """Provides real semantic encoding and HDC operations for associative memory."""
    def __init__(self, dimension: int = 384, model_name: str = "all-MiniLM-L6-v2",
                 encoder_timeout: float = 30.0):
        self.dimension = dimension
        self.model = None
        if SentenceTransformer:
            import threading
            result = [None]
            err = [None]
            def _load():
                try:
                    result[0] = SentenceTransformer(model_name)
                except Exception as e:
                    err[0] = e
            t = threading.Thread(target=_load, daemon=True)
            t.start()
            t.join(timeout=encoder_timeout)
            if t.is_alive():
                print(f"[Substrate] Encoder load timed out after {encoder_timeout}s — using fast fallback")
            elif err[0]:
                print(f"[Substrate] Encoder load failed ({err[0]}) — using fast fallback")
            else:
                self.model = result[0]
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"[Substrate] Encoder ready ({model_name})")
        if self.model is None:
            print(f"[Substrate] Running in fast-fallback mode (hash-based HDC, dim={self.dimension})")

    def encode(self, text: str) -> np.ndarray:
        """Converts raw text into a real semantic vector."""
        if self.model:
            return self.model.encode(text)
        return self._hash_encode(text)

    def _hash_encode(self, text: str) -> np.ndarray:
        """Deterministic hash-based HDC encoding. Same text → same vector."""
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
        """Cosine similarity between two hypervectors. Supports both real-valued unit vectors and bipolar vectors."""
        if self.model:
            # Real embeddings are normalized unit vectors by default from encode()
            return float(np.dot(v1, v2))
        return np.dot(v1, v2) / self.dimension

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
