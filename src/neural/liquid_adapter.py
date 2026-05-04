import numpy as np
import typing as t
from .lora import LoRAAdapter

class LiquidAdapter(LoRAAdapter):
    """
    A specialized LoRA adapter where the B matrix evolves dynamically (Liquid weights).
    Formula: h = x(W_0 + A * B_liquid)
    """
    def __init__(self, adapter_id: str, rank: int, dim_in: int, dim_out: int, tau: float = 0.5):
        super().__init__(adapter_id, rank, dim_in, dim_out)
        self.tau = tau
        self.B_velocity = np.zeros_like(self.B)
        self.base_rank = rank

    def evolve_weights(self, x: np.ndarray, dt: float = 0.1):
        """Evolves the B matrix based on input dynamics."""
        # Simple Euler integration for weight evolution: dB/dt = -B/tau + Outer(A^T*x^T, x)
        # We simplify the gradient-like update for simulation
        innovation = np.outer(np.dot(self.A.T, x.T).mean(axis=1), x.mean(axis=0))
        # Ensure innovation matches B's shape (rank, dim_out)
        innovation = innovation[:self.rank, :self.dim_out]
        
        self.B_velocity = (-self.B / self.tau) + innovation
        self.B += self.B_velocity * dt
        
        # Update delta_W
        self.delta_W = np.dot(self.A, self.B)

    def adjust_rank(self, input_complexity: float):
        """Dynamically adjusts rank based on perceived input complexity/variance."""
        # In a real system, this would involve re-allocation or masking
        # For simulation, we modulate the impact factor
        new_rank = int(self.base_rank * (1.0 + input_complexity))
        # Simulated rank shift: we just log it for this prototype
        # print(f"[LiquidAdapter] Rank Adjusted to: {new_rank}")
        pass

    def apply(self, x: np.ndarray, evolve: bool = True) -> np.ndarray:
        """Applies the update and optionally evolves weights."""
        if evolve:
            self.evolve_weights(x)
        return super().apply(x)
