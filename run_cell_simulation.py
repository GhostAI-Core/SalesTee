import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from core.ocap import OCapManager
from core.cell import AgenticCell
from substrate.hdc import HDCSubstrate, AssociativeMemory

def run_simulation():
    print("=== Starting Agentic Data System Simulation ===\n")
    
    # 1. Initialize Management Layers
    ocap_mgr = OCapManager()
    hdc_substrate = HDCSubstrate(dimension=1000) # Smaller for simulation
    discovery_service = AssociativeMemory(hdc_substrate)
    
    # 2. Create Capabilities (Symbolic Governance)
    print("[Governance] Minting capabilities for 'system' actions...")
    cap_continue = ocap_mgr.mint("system", "continue_operation")
    cap_rest = ocap_mgr.mint("system", "emergency_rest")
    
    # 3. Initialize Agentic Cells
    print("[Simulation] Creating Agentic Cell: 'Alpha'...")
    cell_alpha = AgenticCell("Alpha", ocap_mgr)
    
    # Grant ONLY continue capability initially
    cell_alpha.grant_capability("action_continue_operation", cap_continue)
    
    # 4. HDC Discovery Setup
    print("[Associative] Registering Cell 'Alpha' in HDC discovery space...")
    alpha_vec = hdc_substrate.random_hypervector()
    discovery_service.store("Cell_Alpha", alpha_vec)
    
    # 5. Run PRA Cycles
    print("\n--- Running PRA Cycles for 'Alpha' ---")
    
    # Cycle 1: Normal operation
    cell_alpha.run_cycle()
    
    # Cycle 2: Force energy low to trigger 'emergency_rest' reasoning
    cell_alpha.state["energy"] = 10
    print(f"[{cell_alpha.cell_id}] ENERGY MANUALLY SET TO {cell_alpha.state['energy']}")
    
    # This should FAIL because 'Alpha' hasn't been granted cap_rest yet
    cell_alpha.run_cycle()
    
    # 6. Attentuate and Grant New Capability
    print("\n[Governance] Granting 'emergency_rest' capability to 'Alpha'...")
    cell_alpha.grant_capability("action_emergency_rest", cap_rest)
    
    # Cycle 3: Now it should succeed
    cell_alpha.run_cycle()
    
    # 7. HDC Discovery Verification
    print("\n--- Testing Associative Discovery ---")
    # Simulate a noisy query for 'Alpha'
    noise1 = hdc_substrate.random_hypervector()
    noise2 = hdc_substrate.random_hypervector()
    noisy_query = hdc_substrate.bundle([alpha_vec, noise1, noise2]) # Mostly Alpha
    
    matches = discovery_service.search(noisy_query, threshold=0.1)
    print(f"[Discovery] Searching for Cell Alpha with noisy query...")
    if not matches:
        # Debug: Print the actual similarity
        sim = hdc_substrate.similarity(noisy_query, alpha_vec)
        print(f"[Debug] No matches above threshold. Actual similarity to Alpha: {sim:.4f}")
    
    for label, sim in matches:
        print(f"Found: {label} (Similarity: {sim:.4f})")

    print("\n[Simulation] History of 'Alpha':")
    for event in cell_alpha.history:
        print(f"  - {event}")

if __name__ == "__main__":
    run_simulation()
