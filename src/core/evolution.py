import typing as t
from .ocap import OCapManager, Capability

class EvolutionBridge:
    """Bridges neural reasoning to symbolic governance."""
    def __init__(self, ocap_manager: OCapManager):
        self.ocap_manager = ocap_manager
        self.global_rules: t.Set[str] = {"energy_must_be_positive"}
        self.proposed_rules: t.List[t.Dict[str, t.Any]] = []

    def propose_rule(self, cell_id: str, rule_type: str, action: str):
        """Neural layer proposes a new behavior or rule adaptation."""
        proposal = {
            "cell_id": cell_id,
            "rule_type": rule_type,
            "action": action,
            "status": "pending"
        }
        self.proposed_rules.append(proposal)
        return proposal

    def vet_proposals(self) -> t.List[t.Dict[str, t.Any]]:
        """A symbolic vetting process that approves or rejects rules."""
        approved = []
        for prop in self.proposed_rules:
            if prop["status"] != "pending": continue
            
            # Governance Logic: Only allow 'efficiency' rules that don't violate global rules
            if prop["rule_type"] == "efficiency":
                prop["status"] = "approved"
                approved.append(prop)
            else:
                prop["status"] = "rejected"
        
        return approved

    def apply_approved_rules(self, approved_list: t.List[t.Dict[str, t.Any]], cells_map: t.Dict[str, t.Any]):
        """Grants new capabilities based on approved rules."""
        for prop in approved_list:
            cell = cells_map.get(prop["cell_id"])
            if cell:
                # Mint and grant the new capability
                new_cap = self.ocap_manager.mint("system", prop["action"])
                cell.grant_capability(f"action_{prop['action']}", new_cap)
                cell.history.append(f"evolved: adopted {prop['action']} via {prop['rule_type']} rule")
