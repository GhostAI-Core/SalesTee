import numpy as np
import typing as t

class NCALattice:
    """Lattice of cells that self-organize via local neural update rules."""
    def __init__(self, rows: int, cols: int, state_dim: int = 4):
        self.rows = rows
        self.cols = cols
        self.state_dim = state_dim
        # State: [energy, connection_strength, status_bit, type_id]
        self.grid = np.random.random((rows, cols, state_dim))

    def step(self):
        """One step of the NCA: Local kernels define the next state."""
        new_grid = self.grid.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                # Neighbors (Moore neighborhood)
                neighbors = self.grid[max(0, r-1):min(self.rows, r+2), 
                                     max(0, c-1):min(self.cols, c+2)]
                
                # Rule 1: Energy Diffusion (Local Smoothing)
                avg_energy = np.mean(neighbors[:, :, 0])
                new_grid[r, c, 0] = 0.9 * self.grid[r, c, 0] + 0.1 * avg_energy
                
                # Rule 2: Connection Crystallization
                # If neighbors have high energy, increase connection strength
                if avg_energy > 0.7:
                    new_grid[r, c, 1] = min(1.0, self.grid[r, c, 1] + 0.05)
                
                # Rule 3: Self-Repair
                # If status_bit is low (dead cell), neighbors can 'revive' it
                if self.grid[r, c, 2] < 0.2 and avg_energy > 0.6:
                    new_grid[r, c, 2] = 0.5 # Revived
                    
        self.grid = new_grid

    def get_structure_stats(self) -> t.Dict[str, float]:
        """Calculates global emergence metrics."""
        return {
            "avg_connectivity": float(np.mean(self.grid[:, :, 1])),
            "active_cells": float(np.sum(self.grid[:, :, 2] > 0.1)),
            "system_entropy": float(-np.sum(self.grid * np.log(self.grid + 1e-9)) / self.grid.size)
        }
