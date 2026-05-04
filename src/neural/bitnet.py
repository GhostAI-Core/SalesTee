import numpy as np
import typing as t

class BitNetb158:
    """Simulates a 1.58-bit (ternary) linear layer from BitNet."""
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights and then quantize to {-1, 0, 1}
        W = np.random.randn(out_features, in_features)
        self.weights = self._quantize_weights(W)
        self.energy_profile = {"fp_mult": 0, "int_add": 0}

    def _quantize_weights(self, W: np.ndarray) -> np.ndarray:
        """Quantizes weights to {-1, 0, 1} using the b1.58 strategy."""
        W_base = np.abs(W).mean()
        if W_base == 0: return np.zeros_like(W)
        W_q = np.round(W / W_base)
        return np.clip(W_q, -1, 1).astype(int)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using only integer addition/subtraction (simulated)."""
        # x is assumed to be quantized/scaled
        # Result = x @ self.weights.T
        # Energy accounting:
        # Instead of FP multiplications, we do additions of x where weight is 1 
        # and subtractions where weight is -1.
        
        num_ops = x.size * self.out_features
        self.energy_profile["int_add"] += num_ops
        
        return np.dot(x, self.weights.T)

    def estimate_energy_savings(self) -> str:
        """Returns a string describing the theoretical energy savings."""
        saved = (1.0 - 0.2) * 100 # Approx 80% reduction for integer logic
        return f"Theoretical Energy Reduction: {saved:.1f}% compared to FP16"
