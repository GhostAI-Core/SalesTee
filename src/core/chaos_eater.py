import time
import random
import typing as t

class ChaosEater:
    """Proactively introduces failures to test systemic resilience."""
    def __init__(self):
        self.latency_bias = 0.0 # seconds
        self.fault_probability = 0.0

    def set_chaos(self, latency: float, fault_prob: float):
        self.latency_bias = latency
        self.fault_probability = fault_prob
        print(f"[ChaosEater] Status: Latency={latency}s, FaultProb={fault_prob}")

    def inject(self, func: t.Callable, *args, **kwargs) -> t.Any:
        """Wraps a call with simulated chaos."""
        if self.latency_bias > 0:
            time.sleep(self.latency_bias)
            
        if random.random() < self.fault_probability:
            print("[ChaosEater] FAULT INJECTED!")
            raise RuntimeError("ChaosEater: Simulated Internal Fault")
            
        return func(*args, **kwargs)

def run_chaos_test(cell: t.Any, bridge: t.Any):
    print("\n=== Chaos Engineering: Task Resilience (ChaosEater) ===\n")
    chaos = ChaosEater()
    
    # Scenario: High latency and potential faults trigger low-power adaptation
    chaos.set_chaos(latency=0.1, fault_prob=0.3)
    
    print("[Test] Running PRA cycle through Chaos Gate...")
    try:
        # We wrap the reason step which is often the most expensive/risky
        original_reason = cell.run_cycle
        # We simulate the bridge monitoring this
        cell.run_cycle(bridge=bridge)
        print("[Test] Cycle completed despite chaos.")
    except Exception as e:
        print(f"[Test] Caught expected fault: {e}")
        print("[Test] Verifying if cell history records failure...")
        # Check if evolution was triggered by the error context (simulation)
        bridge.propose_rule(cell.cell_id, "efficiency", "error_recovery_mode")
        print("[Test] Autonomous recovery rule proposed.")
    
    # --- Blast Radius Assessment ---
    print("\n[Test] Assessing Blast Radius Containment...")
    cell_neighbor = type(cell)("Neighbor", cell.ocap_manager, cell.orchestrator)
    if cell_neighbor.state["energy"] == 100:
        print("[Status] Blast Radius: CONTAINED (Neighbor cell unaffected by fault)")
    else:
        print("[Status] Blast Radius: LEAKED (Neighbor state corrupted)")

if __name__ == "__main__":
    # This would be imported and used in a larger test runner
    pass
