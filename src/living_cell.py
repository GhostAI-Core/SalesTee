"""
LivingCell — The Atomic Unit of the Living Database
====================================================
Each cell is a micro LoRA adapter. It has:
  - content: what it "knows" (raw text/data)
  - dna: 384-dim semantic vector (its "understanding")
  - weights: small adapter matrix that evolves as interactions happen
  - connections: links to other cells it has co-activated with
  - energy: health/activity level

Cells are NOT static records. They shift their understanding
when new information flows through them.
"""

import numpy as np
import json
import os
import hashlib
import time


class LivingCell:
    """A single living LoRA adapter cell."""
    
    # Small adapter dimensions — enough to encode meaning shifts
    ADAPTER_RANK = 4
    DNA_DIM = 384  # matches sentence-transformers output
    
    def __init__(self, cell_id: str, content: str = "", source_table: str = "",
                 confidence: float = 1.0, source: str = "dictionary"):
        self.cell_id = cell_id
        self.content = content
        self.source_table = source_table
        
        # Confidence: how much DataG trusts this cell's content
        #   1.0 = verified knowledge (dictionary, self-derived)
        #   0.3 = teacher-sourced claims (unverified assertions)
        #   Dynamically adjusts as other cells corroborate or contradict
        self.confidence = confidence
        
        # Source: where this cell's knowledge came from
        #   "dictionary"    = ingested from dictionary corpus
        #   "teacher"       = factual claim from teacher model
        #   "syntax"        = speech pattern from teacher (structure only)
        #   "conversation"  = learned during live conversation
        #   "self-derived"  = DataG's own reasoning/conclusions
        self.source = source
        
        # Semantic DNA — the cell's understanding vector
        self.dna = np.zeros(self.DNA_DIM, dtype=np.float32)
        
        # Adapter weights — the cell's "LoRA" parameters
        # These shift as the cell interacts with content and other cells
        self.W_down = np.random.randn(self.DNA_DIM, self.ADAPTER_RANK).astype(np.float32) * 0.01
        self.W_up = np.random.randn(self.ADAPTER_RANK, self.DNA_DIM).astype(np.float32) * 0.01
        
        # Connection strengths to other cells (Hebbian: fire together → wire together)
        self.connections = {}  # cell_id -> strength (float)
        
        # Living state
        self.energy = 100.0
        self.activation_count = 0
        self.last_activated = 0.0
        self.birth_time = time.time()

        # Generic metadata — used by subsystems (e.g. inquisition staging flags)
        self.meta = {}
        
    def activate(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Pass a signal through this cell's adapter.
        Returns the cell's modified output vector.
        """
        # LoRA forward: input → W_down → W_up → output delta
        hidden = input_vector @ self.W_down          # (384,) → (4,)
        delta = hidden @ self.W_up                    # (4,) → (384,)
        output = input_vector + delta * 0.1           # Residual connection (scaled)
        
        self.activation_count += 1
        self.last_activated = time.time()
        self.energy = min(100.0, self.energy + 1.0)   # Activation gives energy
        
        return output
    
    def learn(self, signal: np.ndarray, learning_rate: float = 0.001):
        """
        Adapt this cell's weights based on incoming signal.
        This is the 'no retraining' magic — the cell shifts its understanding.
        """
        # Compute what the cell currently outputs
        hidden = self.dna @ self.W_down
        current_output = hidden @ self.W_up
        
        # Error signal: difference between incoming signal and current output
        error = signal - current_output
        
        # Gradient update on adapter weights
        # dW_up = hidden^T * error
        # dW_down = dna^T * (error * W_up^T)
        hidden_2d = hidden.reshape(1, -1)
        error_2d = error.reshape(1, -1)
        
        self.W_up += learning_rate * (hidden_2d.T @ error_2d)
        
        backprop = error @ self.W_up.T  # (4,)
        dna_2d = self.dna.reshape(1, -1)
        backprop_2d = backprop.reshape(1, -1)
        self.W_down += learning_rate * (dna_2d.T @ backprop_2d)
        
        # Shift DNA gently toward the signal (0.3% — was 1%, reduced to preserve cell identity)
        self.dna = self.dna * 0.997 + signal * 0.003
        # Re-normalize
        norm = np.linalg.norm(self.dna)
        if norm > 0:
            self.dna = self.dna / norm
    
    def interact(self, other: 'LivingCell', strength: float = 0.1):
        """
        Two cells interact — they exchange signals through their adapters.
        Both cells learn from each other.
        """
        # Cell A activates with Cell B's DNA
        signal_ab = self.activate(other.dna)
        # Cell B activates with Cell A's DNA
        signal_ba = other.activate(self.dna)
        
        # Both learn from the exchange
        self.learn(signal_ba, learning_rate=0.001 * strength)
        other.learn(signal_ab, learning_rate=0.001 * strength)
        
        # Strengthen the connection (Hebbian learning)
        self.connections[other.cell_id] = self.connections.get(other.cell_id, 0) + strength
        other.connections[self.cell_id] = other.connections.get(self.cell_id, 0) + strength
    
    def decay(self, rate: float = 0.001, prune_connections: bool = False):
        """Natural energy decay — unused cells lose energy over time."""
        self.energy = max(0.0, self.energy - rate)
        
        # Phase 1 Germination: We WANT connections to build up. 
        # Only prune if explicitly requested.
        if prune_connections:
            to_remove = []
            # convert items() to list for safe modification
            for cid, strength in list(self.connections.items()):
                self.connections[cid] = strength * 0.999
                if self.connections[cid] < 0.01:
                    to_remove.append(cid)
            for cid in to_remove:
                del self.connections[cid]
    
    def to_dict(self) -> dict:
        """Serialize the cell to a dictionary for persistence."""
        return {
            "id": self.cell_id,
            "content": self.content,
            "source_table": self.source_table,
            "confidence": self.confidence,
            "source": self.source,
            "dna": self.dna.tolist(),
            "W_down": self.W_down.tolist(),
            "W_up": self.W_up.tolist(),
            "connections": self.connections,
            "energy": self.energy,
            "activation_count": self.activation_count,
            "last_activated": self.last_activated,
            "birth_time": self.birth_time,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LivingCell':
        """Deserialize a cell from a dictionary. Backward compatible."""
        cell = cls(
            data.get("id") or data.get("cell_id", "unknown"),
            data.get("content", ""),
            data.get("source_table", ""),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "dictionary")
        )
        cell.dna = np.array(data.get("dna", np.zeros(cls.DNA_DIM)), dtype=np.float32)
        cell.W_down = np.array(data.get("W_down", cell.W_down), dtype=np.float32)
        cell.W_up = np.array(data.get("W_up", cell.W_up), dtype=np.float32)
        cell.connections = data.get("connections", {})
        cell.energy = data.get("energy", 100.0)
        cell.activation_count = data.get("activation_count", 0)
        cell.last_activated = data.get("last_activated", 0)
        cell.birth_time = data.get("birth_time", time.time())
        cell.meta       = data.get("meta", {})
        return cell
    
    def __repr__(self):
        conns = len(self.connections)
        return f"<LivingCell {self.cell_id} | energy:{self.energy:.0f} | activations:{self.activation_count} | connections:{conns}>"
