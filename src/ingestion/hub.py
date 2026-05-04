import typing as t
import numpy as np
from core.ocap import OCapManager
from core.cell import AgenticCell
from substrate.hdc import HDCSubstrate
from substrate.mhn import ModernHopfieldNetwork
from neural.router import SemanticRouter
from neural.orchestrator import orchestrator_singleton

class IntelligentIngestionHub:
    """Assimilates raw data into living agentic cells with identities and semantic DNA."""
    def __init__(self, ocap_mgr: OCapManager, hdc: HDCSubstrate, router: SemanticRouter, mhn: ModernHopfieldNetwork):
        self.ocap_mgr = ocap_mgr
        self.hdc = hdc
        self.router = router
        self.mhn = mhn
        self.orchestrator = orchestrator_singleton

    def assimilate(self, raw_data: t.Union[t.Dict[str, t.Any], t.List[t.Dict[str, t.Any]]]) -> t.Union[AgenticCell, t.List[AgenticCell]]:
        """Converts a raw record (or list of records) into living Data Organism cell(s)."""
        if isinstance(raw_data, list):
            return [self._assimilate_single(record) for record in raw_data]
        return self._assimilate_single(raw_data)

    def _assimilate_single(self, raw_data: t.Dict[str, t.Any]) -> AgenticCell:
        record_id = raw_data.get("id", f"record_{np.random.randint(1000000)}")
        
        # 1. Identity Attribution (OCap)
        # Grant restricted manifest: only self-maintenance and basic query cooperation
        manifest = ["action_continue_operation", "action_emergency_rest"]
        
        # Register default adapter for the cell
        from neural.lora import LoRAAdapter
        adapter_id = f"adapter_{record_id}"
        self.orchestrator.register_adapter(LoRAAdapter(adapter_id, rank=8, dim_in=512, dim_out=512))
        
        cell = AgenticCell(record_id, self.ocap_mgr, self.orchestrator, adapter_id=adapter_id)
        for action in manifest:
            cell.grant_capability(action, self.ocap_mgr.mint("system", action))
        
        # 2. Semantic Encoding (Real Neural Embedding)
        # Encode keys/values into a real semantic hypervector
        content_str = str(raw_data)
        semantic_dna = self.hdc.encode(content_str)
        cell.state["semantic_dna"] = semantic_dna.tolist()
        self.mhn.store(record_id, semantic_dna)
        
        # 3. Intellect Attachment (Semantic Routing)
        # Route to specialized LoRA based on data shape/content
        intellect_tag = self.router.route(content_str)
        if intellect_tag:
            cell.adapter_id = intellect_tag
            cell.history.append(f"assimilated with intellect {intellect_tag}")
        
        # 4. Impact Contract
        cell.state["impact"] = raw_data.get("impact", "nominal")
        
        return cell
