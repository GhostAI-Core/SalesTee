import typing as t
import collections
from .lora import LoRAAdapter

class UnifiedMemoryPool:
    """Simulates a paged memory pool for adapters and KV caches."""
    def __init__(self, total_pages: int, page_size: int):
        self.total_pages = total_pages
        self.page_size = page_size
        self.free_pages = list(range(total_pages))
        self.active_pages: t.Dict[str, t.List[int]] = {}

    def allocate(self, resource_id: str, num_pages: int) -> bool:
        if len(self.free_pages) < num_pages:
            return False
        pages = [self.free_pages.pop() for _ in range(num_pages)]
        self.active_pages[resource_id] = pages
        return True

    def evict(self, resource_id: str):
        if resource_id in self.active_pages:
            pages = self.active_pages.pop(resource_id)
            self.free_pages.extend(pages)

class SLoRAOrchestrator:
    """Manages adapter loading and inference batching."""
    def __init__(self, base_model_weights: int = 1000):
        self.base_model_weights = base_model_weights # Simulated size
        self.memory_pool = UnifiedMemoryPool(total_pages=1024, page_size=1)
        self.adapter_registry: t.Dict[str, LoRAAdapter] = {}
        self.gpu_cache: t.Set[str] = set()
        self.lru_queue = collections.deque()

    def register_adapter(self, adapter: LoRAAdapter):
        self.adapter_registry[adapter.adapter_id] = adapter

    def _ensure_loaded(self, adapter_id: str):
        """Mock loading adapter weights from RAM to GPU using paging."""
        if adapter_id in self.gpu_cache:
            # Update LRU
            self.lru_queue.remove(adapter_id)
            self.lru_queue.append(adapter_id)
            return

        # Allocate space in memory pool
        num_pages_needed = 4 # Simulated size for 1 adapter
        while not self.memory_pool.allocate(adapter_id, num_pages_needed):
            if not self.lru_queue:
                raise MemoryError("Orchestrator: Out of simulated GPU memory!")
            to_evict = self.lru_queue.popleft()
            self.gpu_cache.remove(to_evict)
            self.memory_pool.evict(to_evict)

        self.gpu_cache.add(adapter_id)
        self.lru_queue.append(adapter_id)

    def run_inference(self, adapter_id: str, input_data: t.Any) -> str:
        """Simulates LoRA-backed reasoning."""
        self._ensure_loaded(adapter_id)
        adapter = self.adapter_registry[adapter_id]
        
        # Simulated logic: input_data might be environment state
        # In a real system, the adapter's output would be sampled
        # For simulation, we use the adapter ID to influence the result
        if "energy" in input_data and input_data["energy"] < 25:
            return "emergency_rest"
        
        # Hash based variety for different adapters
        if hash(adapter_id) % 3 == 0:
            return "explore_surroundings"
        return "continue_operation"

orchestrator_singleton = SLoRAOrchestrator()
