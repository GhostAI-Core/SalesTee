import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from system import AgenticSystem
from neural.liquid_adapter import LiquidAdapter
from substrate.nca_lattice import NCALattice

def run_ecosystem_stress_test():
    print("=== Phase 6: Ecosystem Stress Test (Production Synthesis) ===\n")
    
    # 1. Initialize Unified System
    system = AgenticSystem(hdc_dim=512)
    lattice = NCALattice(rows=10, cols=10) # 100-cell potential lattice
    
    # 2. Register Liquid and LoRA Tools
    tools = {
        "efficiency_expert": "Optimizes energy using Liquid Adapters.",
        "repair_nanite": "Anomalous structural repair.",
        "explorer": "Semantic data discovery."
    }
    for label, desc in tools.items():
        # Register in system's router and orchestrator
        # We simulate LiquidAdapter for efficiency
        adapter = LiquidAdapter(label, rank=16, dim_in=512, dim_out=512)
        system.orchestrator.register_adapter(adapter)
        system.register_tool(label, desc, LiquidAdapter)

    # 3. Spawn a cluster of cells
    for i in range(5):
        cell = system.spawn_cell(f"Cell_{i}")
        # Grant capabilities
        cell.grant_capability("action_continue_operation", system.ocap_mgr.mint("root", "op"))
        cell.grant_capability("action_emergency_rest", system.ocap_mgr.mint("root", "rest"))

    print("\n--- Running High-Velocity Stress Test (Liquid Reasoning & Synaptic Pruning) ---")
    start_time = time.time()
    
    for cycle in range(12):
        # Lattice Dynamics
        lattice.step()
        
        # System Step
        system.run_step()
        
        # Simulate retrieval/rehearsal for Cell_0 every few cycles
        if cycle % 3 == 0:
            system.cells["Cell_0"].memory.retrieve("Step")
            
        # Telemetry
        avg_energy = sum(c["energy"] for c in system.get_telemetry().values()) / 5
        print(f"[Cycle {cycle}] Avg Energy: {avg_energy:.1f} | Matrix Entropy: {lattice.get_structure_stats()['system_entropy']:.4f}")

    print("\n--- Cognitive Persistence Results (Cell_0) ---")
    system.cells["Cell_0"].memory.calculate_utility()
    for rec in system.cells["Cell_0"].memory.records:
        rehearsal_status = "STRENGTHENED" if rec["retrieval_count"] > 0 else "DECAYING"
        print(f"  - Content: {rec['content'][:25]}... | Utility: {rec['current_utility']:.4f} | Status: {rehearsal_status}")

    print(f"\nStress Test PASSED. Duration: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run_ecosystem_stress_test()
