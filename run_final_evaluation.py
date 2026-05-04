import sys
import os

# Add src and tests to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'tests'))

from tester_associative import benchmark_noise_tolerance
from tester_ocap import run_ocap_audits, test_revocation_logic
from tester_agency import run_agency_metrics
from core.chaos_eater import run_chaos_test
from core.ocap import OCapManager
from core.cell import AgenticCell
from core.evolution import EvolutionBridge
from neural.lora import LoRAAdapter
from neural.orchestrator import orchestrator_singleton

def run_full_evaluation():
    print("====================================================")
    print("AGENTIC DATA SYSTEM: FINAL PHASE 4 EVALUATION")
    print("====================================================\n")
    
    # 1. Security Audits
    test_revocation_logic()
    run_ocap_audits()
    
    # 2. Associative Benchmarks
    print("\n" + "="*50)
    benchmark_noise_tolerance()
    
    # 3. Agency Metrics
    print("\n" + "="*50)
    run_agency_metrics()
    
    # 4. Chaos Engineering
    print("\n" + "="*50)
    ocap_mgr = OCapManager()
    bridge = EvolutionBridge(ocap_mgr)
    adapter = LoRAAdapter("chaos_target", rank=8, dim_in=128, dim_out=128)
    orchestrator_singleton.register_adapter(adapter)
    cell = AgenticCell("Chaos_Cell", ocap_mgr, orchestrator_singleton, adapter_id="chaos_target")
    
    # Grant caps so it can actually try to act
    cell.grant_capability("action_continue_operation", ocap_mgr.mint("system", "continue_operation"))
    
    run_chaos_test(cell, bridge)
    
    print("\n====================================================")
    print("EVALUATION COMPLETE")
    print("====================================================")

if __name__ == "__main__":
    run_full_evaluation()
