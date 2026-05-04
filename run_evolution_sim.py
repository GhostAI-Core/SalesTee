import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from core.ocap import OCapManager
from core.cell import AgenticCell
from core.evolution import EvolutionBridge
from neural.lora import LoRAAdapter
from neural.orchestrator import orchestrator_singleton

def run_evolution_sim():
    print("=== Phase 3: Neuro-Symbolic Evolution & Micro-Agents Simulation ===\n")
    
    # 1. Setup Layered Architecture
    ocap_mgr = OCapManager()
    bridge = EvolutionBridge(ocap_mgr)
    
    # 2. Register Cell & Adapter
    adapter_id = "specialized_0"
    adapter = LoRAAdapter(adapter_id, rank=8, dim_in=128, dim_out=128)
    orchestrator_singleton.register_adapter(adapter)
    
    cell_delta = AgenticCell("Delta", ocap_mgr, orchestrator_singleton, adapter_id=adapter_id)
    
    # 3. Grant Initial Capabilities
    cell_delta.grant_capability("action_continue_operation", ocap_mgr.mint("system", "continue_operation"))
    cell_delta.grant_capability("action_emergency_rest", ocap_mgr.mint("system", "emergency_rest"))
    
    # 4. Run Cycles with Evolutionary Feedback
    print("--- Running Cycles with Neuro-Symbolic Bridge ---")
    for i in range(8):
        # Force lower energy to trigger rule proposal
        cell_delta.state["energy"] -= 10
        cell_delta.run_cycle(bridge=bridge)
        
        # Run vetting halfway through and at the end
        if i == 3 or i == 7:
            print(f"\n[Governance] Vetting neural proposals (Cycle {i})...")
            approved = bridge.vet_proposals()
            bridge.apply_approved_rules(approved, {"Delta": cell_delta})
            print(f"[Governance] Approved {len(approved)} rule(s).\n")

    # 5. BitNet Energy Verification
    print("\n--- Micro-Agent Monitoring (BitNet) ---")
    print(cell_delta.monitor.estimate_energy_savings())
    print(f"Ops recorded: {cell_delta.monitor.energy_profile['int_add']} additions")

    # 6. Final State Review
    print(f"\n[Simulation] Final History of {cell_delta.cell_id}:")
    for event in cell_delta.history:
        print(f"  - {event}")

if __name__ == "__main__":
    run_evolution_sim()
