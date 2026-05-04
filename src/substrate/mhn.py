import numpy as np
import typing as t

class ModernHopfieldNetwork:
    """Simulates a Modern Hopfield Network (Dense Associative Memory)."""
    def __init__(self, dimension: int, beta: float = 1.0):
        self.d = dimension
        self.beta = beta
        self.patterns: t.Optional[np.ndarray] = None # Matrix X [d, C]
        self.labels: t.List[str] = []

    def store(self, label: str, pattern: np.ndarray):
        """Stores a pattern in the network."""
        # Ensure pattern is a column vector
        pattern = pattern.reshape(self.d, 1)
        if self.patterns is None:
            self.patterns = pattern
        else:
            self.patterns = np.hstack([self.patterns, pattern])
        self.labels.append(label)

    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """Single-step associative retrieval (log-sum-exp)."""
        if self.patterns is None:
            raise ValueError("MHN: No patterns stored.")
        
        # Softmax over dot products: attention = softmax(beta * X^T * query)
        # We use log-sum-exp implicitly through stable softmax
        dot_products = np.dot(self.patterns.T, query.reshape(self.d, 1))
        
        # Stable softmax implementation
        shifted = self.beta * (dot_products - np.max(dot_products))
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        
        # Reconstruct: output = X * weights
        retrieved = np.dot(self.patterns, weights)
        return retrieved.flatten()

    def find_match(self, query: np.ndarray, threshold: float = 0.5) -> t.Optional[t.Tuple[str, float]]:
        """Retrieves and identifies the best matching label."""
        retrieved = self.retrieve(query)
        
        # Find similarity to all stored patterns
        best_sim = -1.0
        best_label = None
        
        for i in range(len(self.labels)):
            sim = np.dot(retrieved, self.patterns[:, i]) / (np.linalg.norm(retrieved) * np.linalg.norm(self.patterns[:, i]))
            if sim > best_sim:
                best_sim = sim
                best_label = self.labels[i]
        
        return (best_label, best_sim) if best_sim >= threshold else None
