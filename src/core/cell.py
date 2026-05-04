import time
import typing as t
import numpy as np
from .ocap import OCapManager, Capability, Membrane
from neural.orchestrator import SLoRAOrchestrator
from neural.bitnet import BitNetb158
from neural.liquid_context import LiquidContext
from substrate.episodic import EpisodicMemory

class AgenticCell:
    """The fundamental unit of agentic data: sensing, reasoning, and acting."""
    def __init__(self, cell_id: str, ocap_manager: OCapManager, orchestrator: SLoRAOrchestrator, adapter_id: t.Optional[str] = None):
        self.cell_id = cell_id
        self.ocap_manager = ocap_manager
        self.orchestrator = orchestrator
        self.adapter_id = adapter_id or f"adapter_{cell_id}"
        self.state: t.Dict[str, t.Any] = {"status": "living", "energy": 100, "anomalies": 0, "repair_attempts": 0}
        self.history: t.List[str] = []
        self._capabilities: t.Dict[str, Capability] = {}
        # Phase 3: Micro-Agent Monitor (BitNet)
        self.monitor = BitNetb158(in_features=10, out_features=1)
        # Phase 5: Living Components
        self.liquid_store = LiquidContext(size=5, dimension=10)
        self.memory = EpisodicMemory(capacity=10)

    def grant_capability(self, name: str, cap: Capability):
        """赋予单元特定权限"""
        self._capabilities[name] = cap

    def perceive(self) -> t.Dict[str, t.Any]:
        """Sensing stage."""
        # Removed simulated random anomaly generator (Fabrication)
        return {"cell_id": self.cell_id, "state": self.state}

    def reason(self, stimuli: t.Dict[str, t.Any], router: t.Optional[t.Any] = None) -> str:
        """Reasoning stage with Semantic Routing and Liquid State."""
        # Evolve liquid state based on environment stimuli
        stim_vec = np.random.randn(10) # Mock vectorization of stimuli
        liquid_output = self.liquid_store.update(stim_vec)
        
        # Determine adapter via Semantic Router if available
        current_adapter = self.adapter_id
        if router:
            intent_str = f"Context: {stimuli['state']}, LiquidDrift: {self.liquid_store.measure_drift()}"
            routed = router.route(intent_str)
            if routed:
                current_adapter = routed
                self.history.append(f"routed to {current_adapter}")

        self.history.append(f"reasoned via adapter {current_adapter}")
        
        # Execute reasoning
        plan = self.orchestrator.run_inference(current_adapter, stimuli["state"])
        
        # Store in episodic memory
        self.memory.record(f"Step: {plan}, Energy: {self.state['energy']}", importance=0.8)
        
        return plan

    def act(self, plan: str):
        """Action stage of the PRA loop."""
        # Actions are gated by capabilities
        cap_name = f"action_{plan}"
        cap = self._capabilities.get(cap_name)
        
        if not cap or not self.ocap_manager.verify(cap, "system", plan):
            self.history.append(f"failed to act: {plan} (PermissionDenied)")
            return
        
        self.history.append(f"acted: {plan}")
        if plan == "continue_operation":
            self.state["energy"] -= 5
        elif plan == "emergency_rest":
            self.state["energy"] += 50

    def run_cycle(self, bridge: t.Optional[t.Any] = None, router: t.Optional[t.Any] = None):
        """One iteration of the PRA loop."""
        stimuli = self.perceive()
        
        # Check for self-repair triggers
        if self.state["anomalies"] > 3:
            self.repair()
            return

        # If it survived without repair, decay the repair attempts organically
        if self.state.get("repair_attempts", 0) > 0:
            self.state["repair_attempts"] -= 1

        plan = self.reason(stimuli, router=router)
        
        # Evolutionary Proposal Logic
        if bridge and stimuli["state"]["energy"] < 50:
            bridge.propose_rule(self.cell_id, "efficiency", "low_power_optimization")

        self.act(plan)
        print(f"[{self.cell_id}] Cycle Complete. Energy: {self.state['energy']}, Anomalies: {self.state['anomalies']}")

    def repair(self):
        """Self-repair logic for structural adaptation."""
        self.state["repair_attempts"] += 1
        
        if self.state["repair_attempts"] > 3:
            self.history.append("critical failure: organic rest ineffective")
            self.state["status"] = "critical_failure"
            print(f"[{self.cell_id}] CRITICAL ALARM: Repair Threshold Exceeded.")
            return

        self.history.append("triggered self-repair")
        self.state["anomalies"] = 0
        self.state["energy"] = min(self.state["energy"] + 20, 100)
        print(f"[{self.cell_id}] Self-Repair Executed. Attempt {self.state['repair_attempts']}/3")
