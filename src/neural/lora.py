import numpy as np
import typing as t
from dataclasses import dataclass

@dataclass
class LoRAAdapter:
    """Simulates a task-specific adapter with Low-Rank matrices A and B."""
    adapter_id: str
    rank: int
    dim_in: int
    dim_out: int
    
    def __post_init__(self):
        # Initialize A and B matrices
        self.A = np.random.randn(self.dim_in, self.rank) * 0.01
        self.B = np.random.randn(self.rank, self.dim_out) * 0.01
        self.delta_W = np.dot(self.A, self.B)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Applies the low-rank update: x * (A * B)"""
        return np.dot(x, self.delta_W)

class SharedBasisOptimizer:
    """Simulates Joint Compression (JD-Full) by projecting adapters onto a shared basis."""
    def __init__(self, shared_u: np.ndarray, shared_v: np.ndarray):
        self.U = shared_u # [dim_in, basis_rank]
        self.V = shared_v # [basis_rank, dim_out]
        
    def compress_adapter(self, adapter: LoRAAdapter) -> np.ndarray:
        """Approximates adapter update using the shared basis returns a scaling matrix Sigma."""
        # Simple approximation for simulation: project delta_W onto U and V
        # Sigma = U^T * DeltaW * V
        sigma = np.matmul(np.matmul(self.U.T, adapter.delta_W), self.V.T)
        return sigma

    def reconstruct(self, sigma: np.ndarray) -> np.ndarray:
        """Reconstructs DeltaW from stored basis and per-adapter Sigma."""
        return np.matmul(np.matmul(self.U, sigma), self.V)
