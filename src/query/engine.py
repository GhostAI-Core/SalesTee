import typing as t
import numpy as np
from core.cell import AgenticCell
from substrate.mhn import ModernHopfieldNetwork
from substrate.hdc import HDCSubstrate

class AccessIntentEngine:
    """Processes 'ACCESS' intents by triggering associative retrieval."""
    def __init__(self, cells: t.Dict[str, AgenticCell], mhn: ModernHopfieldNetwork, hdc: HDCSubstrate):
        self.cells = cells
        self.mhn = mhn
        self.hdc = hdc

    def access(self, target_goal: str, context: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """Resolves an intent goal by associative pattern matching or metadata lookups."""
        results = {}
        print(f"[Engine] Non-Linear Search: {target_goal}")
        
        # 1. Metadata Query Handling
        if "size" in target_goal.lower() or "count" in target_goal.lower():
            return {"system_metadata": {"total_cells": len(self.cells), "status": "nominal"}}

        # 2. Associative Retrieval (Multi-Match)
        query_vec = self.hdc.encode(target_goal) 
        
        # Search all records via MHN
        # In a real system, MHN would return a superposition, but here we iterate for simulation
        threshold = 0.25 if "fault" in target_goal.lower() or "anomaly" in target_goal.lower() else 0.3
        
        # Find top matches
        matches = []
        for cid, cell in self.cells.items():
            if cell.state.get("semantic_dna") is not None:
                sim = self.hdc.similarity(query_vec, np.array(cell.state["semantic_dna"]))
                if sim >= threshold:
                    matches.append((cid, sim))
        
        matches = sorted(matches, key=lambda x: x[1], reverse=True)[:3] # Top 3

        for cid, sim in matches:
            cell = self.cells.get(cid)
            if cell:
                print(f"[Engine] Associative Hit: {cid} (Sim: {sim:.3f})")
                stimuli = {"state": cell.state, "goal": target_goal, "sim": sim}
                proposal = cell.reason(stimuli)
                
                results[cid] = {
                    "status": "associative_match",
                    "proposal": proposal,
                    "confidence": float(sim)
                }
        
        return results
