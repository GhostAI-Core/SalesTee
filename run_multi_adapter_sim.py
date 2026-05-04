import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from core.ocap import OCapManager
from core.cell import AgenticCell
from neural.lora import LoRAAdapter
from neural.orchestrator import orchestrator_singleton

def run_multi_adapter_sim():
    print("=== Phase 2: Multi-Adapter Orchestration Simulation ===\n")
    
    ocap_mgr = OCapManager()
    num_cells = 50 # Simulate 50 cells with 50 unique adapters
    
    print(f"[Simulation] Registering {num_cells} LoRA adapters with Orchestrator...")
    for i in range(num_cells):
        adapter_id = f"adapter_{i}"
        adapter = LoRAAdapter(adapter_id, rank=8, dim_in=128, dim_out=128)
        orchestrator_singleton.register_adapter(adapter)
    
    cells = []
    print(f"[Simulation] Initializing {num_cells} Agentic Cells...")
    for i in range(num_cells):
        cell_id = f"Cell_{i}"
        cell = AgenticCell(cell_id, ocap_mgr, orchestrator_singleton, adapter_id=f"adapter_{i}")
        
        # Grant capabilities for actions determined by adapters
        cell.grant_capability("action_continue_operation", ocap_mgr.mint("system", "continue_operation"))
        cell.grant_capability("action_emergency_rest", ocap_mgr.mint("system", "emergency_rest"))
        cell.grant_capability("action_explore_surroundings", ocap_mgr.mint("system", "explore_surroundings"))
        
        cells.append(cell)

    print("\n--- Running Parallel PRA Cycles (Simulated Paging) ---")
    # Simulate a few rounds of execution
    for round_num in range(1, 4):
        print(f"\n[Round {round_num}]")
        for cell in cells:
            # Randomly fluctuate energy to trigger different reasoning outputs
            import random
            if random.random() < 0.2:
                cell.state["energy"] = random.randint(5, 20)
            
            cell.run_cycle()

    print("\n--- Resource Consumption ---")
    active_adapters = len(orchestrator_singleton.gpu_cache)
    free_pages = len(orchestrator_singleton.memory_pool.free_pages)
    print(f"[Orchestrator] Active Adapters in Simulated GPU Cache: {active_adapters}")
    print(f"[Orchestrator] Free Paging Slots: {free_pages}")
    print(f"[Orchestrator] LRU Eviction History (Queue Size): {len(orchestrator_singleton.lru_queue)}")

    # Verification: Ensure eviction happened if cache size was reached
    # (Memory pool is 1024 pages, 4 pages per adapter = 256 capacity)
    # With 50 cells, we shouldn't hit eviction yet unless we lower the pool size.
    # Let's verify Alpha's specialized history
    print(f"\n[Simulation] History of {cells[0].cell_id}:")
    for event in cells[0].history:
        print(f"  - {event}")

if __name__ == "__main__":
    run_multi_adapter_sim()
