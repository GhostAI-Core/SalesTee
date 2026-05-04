import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from core.ocap import OCapManager
from core.cell import AgenticCell
from core.evolution import EvolutionBridge
from neural.lora import LoRAAdapter
from neural.orchestrator import orchestrator_singleton
from neural.router import SemanticRouter
from substrate.hdc import HDCSubstrate
from substrate.nca_lattice import NCALattice

def run_organism_sim():
    print("=== Phase 5: Living Data Organism Simulation ===\n")
    
    # 1. Initialize Substrates
    ocap_mgr = OCapManager()
    hdc = HDCSubstrate(dimension=128)
    router = SemanticRouter(hdc)
    lattice = NCALattice(rows=5, cols=5)
    bridge = EvolutionBridge(ocap_mgr)
    
    # 2. Register Specialized Adapters
    adapters = {
        "efficiency_expert": "Optimizes energy consumption and resource allocation.",
        "explorer": "Discovers new relational patterns in the data lattice.",
        "repair_nanite": "Specializes in anomaly detection and structural self-repair."
    }
    for label, desc in adapters.items():
        adapter = LoRAAdapter(label, rank=8, dim_in=128, dim_out=128)
        orchestrator_singleton.register_adapter(adapter)
        router.register_adapter(label, desc)
        
    # 3. Initialize a 'Sample' Cell within the Organism
    cell_id = "Alpha"
    default_adapter_id = f"adapter_{cell_id}"
    orchestrator_singleton.register_adapter(LoRAAdapter(default_adapter_id, rank=8, dim_in=128, dim_out=128))
    
    cell_alpha = AgenticCell(cell_id, ocap_mgr, orchestrator_singleton, adapter_id=default_adapter_id)
    # Grant basic capabilities
    for label in adapters.keys():
        cell_alpha.grant_capability(f"action_continue_operation", ocap_mgr.mint("system", "continue_operation"))
        cell_alpha.grant_capability(f"action_emergency_rest", ocap_mgr.mint("system", "emergency_rest"))

    print("\n--- Running Organism Growth & Adaptation ---")
    for step in range(10):
        # A. NCA Grid Step (Structural Evolution)
        lattice.step()
        stats = lattice.get_structure_stats()
        
        # B. Cell PRA Cycle with Liquid State & Semantic Routing
        # Force energy drop to see routing/evolution
        cell_alpha.state["energy"] = max(0, 100 - step*12)
        cell_alpha.run_cycle(bridge=bridge, router=router)
        
        # C. Telemetry
        print(f"[Step {step}] Matrix Entropy: {stats['system_entropy']:.4f} | Liquid Drift: {cell_alpha.liquid_store.measure_drift():.4f}")
        
    print("\n--- Final Organism State ---")
    print(f"Cell Memory Context: {cell_alpha.memory.get_context()}")
    print(f"Final History Highlights for Alpha:")
    for event in cell_alpha.history[-5:]:
        print(f"  - {event}")

if __name__ == "__main__":
    run_organism_sim()
