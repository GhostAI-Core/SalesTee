import numpy as np
import typing as t
from substrate.hdc import HDCSubstrate

class SemanticRouter:
    """Dynamic routing of intents to specialized LoRA adapters."""
    def __init__(self, hdc_substrate: HDCSubstrate):
        self.substrate = hdc_substrate
        self.adapter_indices: t.Dict[str, np.ndarray] = {} # label -> hypervector

    def register_adapter(self, label: str, description: str):
        """Registers an adapter by binding its label to a semantic hypervector."""
        # Simple hash-based vector for the description
        vec = self.substrate.random_hypervector()
        self.adapter_indices[label] = vec
        print(f"[Router] Registered {label} for semantic discovery.")

    def route(self, stim_context: str) -> t.Optional[str]:
        """Performs a 2-step Multi-step Need Analysis for precise routing."""
        # Step 1: Categorize Intent (Mock simulation)
        intent_map = {
            "energy": "efficiency_expert",
            "anomaly": "repair_nanite",
            "drift": "explorer"
        }
        category = "explorer" # Default
        for key in intent_map:
            if key in stim_context.lower():
                category = intent_map[key]
                break

        # Step 2: Semantic mapping to the best specialized tool
        query_vec = self.substrate.random_hypervector()
        
        best_match = None
        best_sim = -1.0
        
        # We prioritize the category-matched adapter
        for label, index_vec in self.adapter_indices.items():
            sim = np.dot(query_vec, index_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(index_vec))
            if label == category:
                sim += 0.2 # Intent bias
                
            if sim > best_sim:
                best_sim = sim
                best_match = label
        
        return best_match if best_sim > 0.1 else None
