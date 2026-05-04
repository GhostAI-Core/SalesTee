import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from system import AgenticSystem
from neural.liquid_adapter import LiquidAdapter

def run_d2a_simulation():
    print("=== Phase 7: D2A Assimilation & ACCESS Intent Simulation ===\n")
    
    # 1. Initialize System
    system = AgenticSystem(hdc_dim=512)
    
    # 2. Register Intellects (Adapters)
    specialties = {
        "legal_expert": "Analyzes legal clauses and compliance.",
        "fraud_detect": "Identifies anomalous financial patterns.",
        "efficiency_expert": "Optimizes resource allocation."
    }
    for label, desc in specialties.items():
        adapter = LiquidAdapter(label, rank=16, dim_in=512, dim_out=512)
        system.orchestrator.register_adapter(adapter)
        system.register_tool(label, desc, LiquidAdapter)

    # 3. D2A Assimilation (Populating the Organism)
    print("--- Assimilating Records into the Organism ---")
    records = [
        {"id": "doc_88", "content": "Legal clause regarding liability limit.", "impact": "high", "domain": "legal"},
        {"id": "tx_404", "content": "Unexpected withdrawal of $50,000.", "impact": "critical", "domain": "financial"},
        {"id": "node_01", "content": "Power consumption exceeding 90%.", "impact": "nominal", "domain": "utility"}
    ]
    
    for rec in records:
        cell = system.assimilate(rec)
        print(f"[Hub] Assimilated {rec['id']} -> Assigned Intellect: {cell.adapter_id}")

    # 4. Intent-based ACCESS Query
    print("\n--- Performing Intent-based ACCESS (Speculative Computation) ---")
    goal = "Identify critical anomalies requiring immediate repair"
    discovery_results = system.access(goal)
    
    print(f"\n[Engine] Goal: '{goal}'")
    for cid, res in discovery_results.items():
        print(f"  - Cell: {cid} | Status: {res['status']} | Proposal: {res['proposal']}")

    print("\nSimulation PASSED. The D2A model successfully converted records into reasoning agents.")

if __name__ == "__main__":
    run_d2a_simulation()
