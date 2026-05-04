import numpy as np
import typing as t

class LiquidWeight:
    """Simulates a weight that evolves over time based on input dynamics."""
    def __init__(self, dimension: int, tau: float = 1.0):
        self.w = np.random.randn(dimension)
        self.tau = tau # Time constant for 'liquidity'
        self.velocity = np.zeros(dimension)

    def evolve(self, x: np.ndarray, dt: float = 0.1):
        """ODE-inspired weight update: dw/dt = -w/tau + x"""
        # Simple Euler integration of the weight dynamics
        self.velocity = (-self.w / self.tau) + x
        self.w += self.velocity * dt
        return self.w

class LiquidContext:
    """Manages a pool of liquid weights for real-time state adaptation."""
    def __init__(self, size: int, dimension: int):
        self.weights = [LiquidWeight(dimension) for _ in range(size)]
        self.drift_factor = 0.0

    def update(self, input_stream: np.ndarray):
        """Processes a chunk of input to evolve the internal 'liquid' state."""
        outputs = []
        for i, weight in enumerate(self.weights):
            # Each weight evolves slightly differently based on its own tau
            w_new = weight.evolve(input_stream)
            outputs.append(w_new)
        return np.mean(outputs, axis=0)

    def measure_drift(self) -> float:
        """Quantifies how much the 'liquid' state has shifted."""
        velocities = [np.linalg.norm(w.velocity) for w in self.weights]
        self.drift_factor = np.mean(velocities)
        return self.drift_factor
