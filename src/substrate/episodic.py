import time
import typing as t
import numpy as np

class EpisodicMemory:
    """Episodic store with intelligent decay and importance-based pruning."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.records: t.List[t.Dict[str, t.Any]] = [] # [{content, timestamp, importance}]

    def record(self, content: str, importance: float = 1.0):
        """Adds a new memory record with retrieval-based rehearsal."""
        self.records.append({
            "content": content,
            "timestamp": time.time(),
            "importance": importance,
            "retrieval_count": 0
        })
        if len(self.records) > self.capacity:
            self._consolidate()

    def retrieve(self, query: str) -> t.Optional[str]:
        """Simulates memory rehearsal upon retrieval."""
        for rec in self.records:
            if query in rec["content"]:
                rec["retrieval_count"] += 1
                rec["importance"] *= 1.1 # Rehearsal: strength through use
                return rec["content"]
        return None

    def calculate_utility(self):
        """Calculates current utility for all records without pruning."""
        now = time.time()
        tau = 10.0
        for rec in self.records:
            age = now - rec["timestamp"]
            rehearsal_bonus = 1.0 + (rec["retrieval_count"] * 0.2)
            rec["current_utility"] = (rec["importance"] * rehearsal_bonus) * np.exp(-age / tau)

    def _consolidate(self):
        """Prunes memories using Synaptic Pruning (Interference + Decay)."""
        self.calculate_utility()

        # Simplified Interference Logic: If two memories are too similar, prune the lower utility one
        # (This is a O(N^2) mockup for simulation)
        to_remove = set()
        for i in range(len(self.records)):
            for j in range(i + 1, len(self.records)):
                # Mock similarity check
                if self.records[i]["content"][:10] == self.records[j]["content"][:10]:
                    if self.records[i]["current_utility"] > self.records[j]["current_utility"]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

        self.records = [r for idx, r in enumerate(self.records) if idx not in to_remove]

        # Sort and cap
        self.records = sorted(self.records, key=lambda x: x["current_utility"], reverse=True)[:self.capacity]

    def get_context(self) -> str:
        """Retrieves high-utility memory summaries."""
        self._consolidate()
        return " | ".join([r["content"] for r in self.records[:5]])
