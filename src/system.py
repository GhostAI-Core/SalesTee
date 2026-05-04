import typing as t
import os
import json
import glob
import hashlib
import time
import numpy as np
from core.ocap import OCapManager
from core.cell import AgenticCell
from neural.orchestrator import orchestrator_singleton
from neural.router import SemanticRouter
from substrate.hdc import HDCSubstrate
from core.evolution import EvolutionBridge
from substrate.mhn import ModernHopfieldNetwork
from ingestion.hub import IntelligentIngestionHub
from query.engine import AccessIntentEngine
from living_cell import LivingCell

class AgenticSystem:
    """Unified API for the Living Data Organism."""
    
    # Lobe routing: maps mode names to cell ID prefix groups
    # Speech cells are the DOMINANT frame; dictionary cells are TOOLS
    LOBE_CONFIG = {
        "mind": {
            "dominant": ["meth_mind", "meth_creator", "meth_identity"],
            "tools": ["meth_english", "meth_conv"],
            "exclude": ["meth_document", "meth_syntax"],
            "dominant_weight": 10.0,
            "tools_weight": 0.5,
        },
        "speech": {
            "dominant": ["meth_creator", "meth_syntax", "meth_conv", "meth_conversational", "meth_speak", "meth_listen"],
            "tools": ["meth_english"],
            "exclude": ["meth_document"],
            "dominant_weight": 3.0,
            "tools_weight": 1.0,
        },
        "dictionary": {
            "dominant": ["meth_english"],
            "tools": ["meth_syntax", "meth_conv", "meth_conversational", "meth_speak",
                       "meth_creator"],
            "exclude": ["meth_document"],
            "dominant_weight": 2.0,
            "tools_weight": 1.0,
        },
        "document": {
            "dominant": ["meth_document"],
            "tools": [],
            "exclude": [],
            "dominant_weight": 2.0,
            "tools_weight": 1.0,
        },
        "reasoning": {
            "dominant": ["meth_mathematical", "meth_code", "meth_logic", "meth_algebra",
                         "meth_calculus", "meth_statistics", "meth_geometry", "meth_general"],
            "tools": ["meth_english", "meth_syntax", "meth_creator"],
            "exclude": ["meth_document"],
            "dominant_weight": 2.0,
            "tools_weight": 1.0,
        },
        "code": {
            "dominant": ["meth_code", "meth_architect", "meth_synthesis"],
            "tools":    ["meth_conv_tech"],
            "exclude":  ["meth_document", "meth_english"],
            "dominant_weight": 3.0,
            "tools_weight": 0.5,
        },
    }
    
    def __init__(self, hdc_dim: int = 384, slim: bool = False, skip_cells: bool = False):
        self._slim_mode = slim
        self._skip_cells = skip_cells
        self.ocap_mgr = OCapManager()
        self.hdc = HDCSubstrate(dimension=hdc_dim)
        actual_dim = self.hdc.dimension

        self.router = SemanticRouter(self.hdc)
        self.orchestrator = orchestrator_singleton
        self.bridge = EvolutionBridge(self.ocap_mgr)
        self.mhn = ModernHopfieldNetwork(dimension=actual_dim)
        self.cells: t.Dict[str, AgenticCell] = {}

        # [PHASE 1: ASSIMILATION] Living methodology cells
        self.methodology_cells: t.Dict[str, LivingCell] = {}
        self._assimilation_log: t.List[float] = []
        self._methodology_interactions = 0

        # [LOBE INDEX] Per-mode cell caches — rebuilt on boot and when cells change
        self._lobe_index: t.Dict[str, t.Dict[str, t.List[LivingCell]]] = {}
        self._lobe_dirty = True  # Flag to rebuild index

        # D2A Components
        self.hub = IntelligentIngestionHub(self.ocap_mgr, self.hdc, self.router, self.mhn)
        self.engine = AccessIntentEngine(self.cells, self.mhn, self.hdc)

        # Decentralized Storage Layer
        os.makedirs("data_store/cells", exist_ok=True)
        os.makedirs("data_store/methodology", exist_ok=True)
        if not self._skip_cells:
            self._load_persisted_cells()
        else:
            print("[System] skip_cells=True: skipping AgenticCell load (diagnostic mode).")
        self._load_methodology_cells(slim=self._slim_mode)
        self._rebuild_lobe_index()

    def _get_shard_path(self, cell_id: str) -> str:
        h = hashlib.md5(cell_id.encode()).hexdigest()[:2]
        shard_dir = os.path.join("data_store/cells", h)
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"{cell_id}.json")

    def _find_cell_file(self, cell_id: str) -> str:
        shard_path = self._get_shard_path(cell_id)
        if os.path.exists(shard_path):
            return shard_path
        flat_path = os.path.join("data_store/cells", f"{cell_id}.json")
        if os.path.exists(flat_path):
            return flat_path
        return None

    def _load_persisted_cells(self):
        """
        Manifest-Based Tiered Rehydration:
        Reads hot_manifest.json to load only cells with neural DNA.
        """
        import numpy as np
        
        manifest_path = "data_store/hot_manifest.json"
        
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            hot_files = manifest.get("hot_cells", [])
            print(f"[System] Loading {len(hot_files):,} HOT cells from manifest...")
            
            hot_loaded = 0
            for entry in hot_files:
                filepath = entry.get("path")
                if not filepath or not os.path.exists(filepath):
                    continue
                try:
                    with open(filepath, 'r') as f:
                        data_wrapper = json.load(f)
                    
                    cell_id = data_wrapper["id"]
                    saved_state = data_wrapper.get("state", {})
                    
                    from neural.lora import LoRAAdapter
                    adapter_id = f"adapter_{cell_id}"
                    self.orchestrator.register_adapter(LoRAAdapter(adapter_id, rank=8, dim_in=512, dim_out=512))
                    
                    cell = AgenticCell(cell_id, self.ocap_mgr, self.orchestrator, adapter_id=adapter_id)
                    cell.state["energy"] = saved_state.get("energy", 100)
                    cell.state["anomalies"] = saved_state.get("anomalies", 0)
                    cell.state["repair_attempts"] = saved_state.get("repair_attempts", 0)
                    cell.state["status"] = saved_state.get("status", "living")
                    cell.state["semantic_dna"] = saved_state.get("semantic_dna")
                    cell.raw_data = data_wrapper.get("raw_data", {})
                    
                    self.cells[cell_id] = cell
                    if cell.state["semantic_dna"]:
                        self.mhn.store(cell_id, np.array(cell.state["semantic_dna"]))
                    hot_loaded += 1
                except Exception as e:
                    pass
            
            total_cells = manifest.get("total_cells", hot_loaded)
            print(f"[System] Rehydration complete:")
            print(f"  → HOT (in-memory + neural search): {hot_loaded:,} cells")
            print(f"  → TOTAL living cells:               {total_cells:,}")
        else:
            print("[System] No manifest found. Building hot_manifest.json (first boot)...")
            self._build_manifest()
            self._load_persisted_cells()

    def _build_manifest(self):
        shard_files = glob.glob("data_store/cells/*/*.json")
        flat_files = glob.glob("data_store/cells/*.json")
        all_files = shard_files + flat_files
        
        hot_cells = []
        total_scanned = 0
        
        print(f"[Manifest] Scanning {len(all_files):,} files...")
        
        for filepath in all_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                state = data.get("state", {})
                if state.get("semantic_dna") is not None:
                    hot_cells.append({"id": data["id"], "path": filepath})
                
                total_scanned += 1
                if total_scanned % 200000 == 0:
                    print(f"[Manifest] Scanned {total_scanned:,} / {len(all_files):,}...")
            except:
                total_scanned += 1
        
        manifest = {
            "hot_cells": hot_cells,
            "total_cells": total_scanned,
            "hot_count": len(hot_cells)
        }
        
        with open("data_store/hot_manifest.json", "w") as f:
            json.dump(manifest, f)
        
        print(f"[Manifest] Built: {len(hot_cells):,} HOT / {total_scanned:,} total")

    def get_cell(self, cell_id: str):
        if cell_id in self.cells:
            return self.cells[cell_id]
        filepath = self._find_cell_file(cell_id)
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data_wrapper = json.load(f)
                return data_wrapper
            except:
                pass
        return None

    def search_cold(self, keyword: str, limit: int = 20) -> list:
        results = []
        for cell_id in self.cells:
            if keyword.lower() in cell_id.lower():
                results.append(cell_id)
                if len(results) >= limit:
                    break
        return results

    def spawn_cell(self, cell_id: str) -> AgenticCell:
        from neural.lora import LoRAAdapter
        adapter_id = f"adapter_{cell_id}"
        self.orchestrator.register_adapter(LoRAAdapter(adapter_id, rank=8, dim_in=512, dim_out=512))
        cell = AgenticCell(cell_id, self.ocap_mgr, self.orchestrator, adapter_id=adapter_id)
        self.cells[cell_id] = cell
        return cell

    def assimilate(self, raw_data: t.Union[t.Dict[str, t.Any], t.List[t.Dict[str, t.Any]]]) -> t.Union[AgenticCell, t.List[AgenticCell]]:
        if isinstance(raw_data, dict) and raw_data.get("type") == "neural_methodology":
            self._assimilate_methodology(raw_data)
        
        cells = self.hub.assimilate(raw_data)
        
        def _persist(c: AgenticCell, raw: t.Dict):
            c.raw_data = raw
            safe_state = {
                "energy": c.state.get("energy", 100),
                "anomalies": c.state.get("anomalies", 0),
                "repair_attempts": c.state.get("repair_attempts", 0),
                "status": c.state.get("status", "living"),
                "semantic_dna": c.state.get("semantic_dna")
            }
            shard_path = self._get_shard_path(c.cell_id)
            with open(shard_path, "w") as f:
                json.dump({"id": c.cell_id, "state": safe_state, "raw_data": raw}, f)
            self.cells[c.cell_id] = c

        if isinstance(cells, list):
            for cell, raw in zip(cells, raw_data):
                _persist(cell, raw)
            return cells
        else:
            _persist(cells, raw_data)
            return cells
    
    def _assimilate_methodology(self, payload: dict):
        data = payload.get("data", {})
        meta = payload.get("metadata", {})
        
        instruction = data.get("instruction", "")
        output = data.get("final_output", "")
        lobe = meta.get("active_lora", "unknown")
        
        if not instruction or not output:
            return
        
        content_hash = hashlib.md5(f"{instruction}{output}".encode()).hexdigest()[:12]
        cell_id = f"meth_{lobe}_{content_hash}"
        
        if cell_id in self.methodology_cells:
            return
        
        combined_text = f"{instruction} {output}"
        dna_vector = self.hdc.encode(combined_text)
        
        cell = LivingCell(cell_id, content=combined_text, source_table="neural_methodology")
        cell.dna = dna_vector / max(np.linalg.norm(dna_vector), 1e-8)
        
        self.methodology_cells[cell_id] = cell
        self._persist_methodology_cell(cell)
        
        print(f"[Assimilation] New methodology cell: {cell_id} | Lobe: {lobe} | Total: {len(self.methodology_cells)}")
    
    def _persist_methodology_cell(self, cell: LivingCell):
        path = os.path.join("data_store/methodology", f"{cell.cell_id}.json")
        with open(path, "w") as f:
            json.dump(cell.to_dict(), f)
    
    def _load_methodology_cells(self, slim: bool = False):
        """
        Load persisted methodology cells from disk.
        slim=True: skip meth_english_* (86k dictionary cells) — loads in seconds instead of minutes.
        slim=False: load everything (original behaviour).
        """
        meth_dir = "data_store/methodology"
        if not os.path.isdir(meth_dir):
            return

        files = glob.glob(os.path.join(meth_dir, "meth_*.json"))
        loaded = 0
        skipped = 0
        for filepath in files:
            if slim:
                fname = os.path.basename(filepath)
                # Dictionary cells are the bulk — skip them in slim mode
                if fname.startswith("meth_english_"):
                    skipped += 1
                    continue
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                cell = LivingCell.from_dict(data)
                self.methodology_cells[cell.cell_id] = cell
                loaded += 1
            except Exception:
                pass

        if slim and skipped:
            print(f"[Assimilation] Slim mode: loaded {loaded:,} cells, skipped {skipped:,} dictionary cells")
        elif loaded > 0:
            print(f"[Assimilation] Loaded {loaded:,} methodology cells from disk")

    def access(self, goal: str, context: t.Optional[t.Dict] = None) -> t.Dict[str, t.Any]:
        return self.engine.access(goal, context or {})

    def register_tool(self, label: str, description: str, adapter_type: t.Any):
        self.router.register_adapter(label, description)

    def run_step(self):
        for cell in list(self.cells.values()):
            if hasattr(cell, 'run_cycle'):
                cell.run_cycle(bridge=self.bridge, router=self.router)
        self._methodology_interaction_step()
    
    def _methodology_interaction_step(self):
        """UNCAGED: Multiple anchors, threshold-based neighbor count + MULTI-LOBE BLEED."""
        cells = list(self.methodology_cells.values())
        if len(cells) < 2:
            return
        
        # ── 1. STANDARD INTRA-LOBE WIRING (The Dense Engine) ──
        # Scale anchors to population size: square root balance
        n_anchors = max(5, int(np.sqrt(len(cells))))
        anchor_indices = np.random.choice(len(cells), size=n_anchors, replace=False)
        
        # Vectorized similarity for all cells
        dna_matrix = np.array([c.dna for c in cells], dtype=np.float32)
        
        for anchor_idx in anchor_indices:
            anchor = cells[anchor_idx]
            sims = dna_matrix @ anchor.dna
            sims[anchor_idx] = -1.0  # Exclude self
            
            # Standard interaction threshold
            neighbor_mask = sims > 0.15
            neighbor_indices = np.where(neighbor_mask)[0]
            
            if len(neighbor_indices) > 20:
                top_n = neighbor_indices[np.argsort(sims[neighbor_indices])[-20:]]
                neighbor_indices = top_n
            
            for n_idx in neighbor_indices:
                sim = float(sims[n_idx])
                anchor.interact(cells[n_idx], strength=sim)
                self._methodology_interactions += 1

        # ── 2. MULTI-LOBE BLEED (The Synaptic Bridge) ──
        # Force weak connections between high-energy Logic and Speech cells
        if not self._lobe_dirty and "reasoning" in self._lobe_index and "speech" in self._lobe_index:
            reasoning_cells = self._lobe_index["reasoning"]["dominant"]
            speech_cells = self._lobe_index["speech"]["dominant"]
            
            if reasoning_cells and speech_cells:
                # Isolate the most active (high energy) cells from both lobes
                r_candidates = sorted(reasoning_cells, key=lambda c: c.energy, reverse=True)[:15]
                s_candidates = sorted(speech_cells, key=lambda c: c.energy, reverse=True)[:15]
                
                for r_cell in r_candidates:
                    for s_cell in s_candidates:
                        # Calculate direct semantic overlap
                        sim = float(np.dot(r_cell.dna, s_cell.dna))
                        
                        # Lower threshold specifically for cross-lobe bleeding (0.05 instead of 0.15)
                        if sim > 0.05:
                            # Weak Hebbian connection: They fire together, but scaled down 
                            # so speech doesn't overpower logic, and logic doesn't crush speech.
                            r_cell.interact(s_cell, strength=sim * 0.4)
                            self._methodology_interactions += 1
        
        # ── 3. ENERGY DECAY ──
        # Decay ENERGY only — we disabled global connection pruning 
        # so cross-lobe connections can successfully germinate and grow
        for cell in cells:
            cell.decay(rate=0.0003, prune_connections=False)
    
    def _persist_all_methodology(self):
        for cell in self.methodology_cells.values():
            self._persist_methodology_cell(cell)
    
    def get_assimilation_status(self) -> dict:
        cells = list(self.methodology_cells.values())
        n_cells = len(cells)
        
        total_connections = sum(len(c.connections) for c in cells)
        avg_connections = total_connections / max(n_cells, 1)
        avg_activations = sum(c.activation_count for c in cells) / max(n_cells, 1)
        avg_energy = sum(c.energy for c in cells) / max(n_cells, 1)
        
        phase = "DORMANT"
        phase_2_ready = False
        phase_2_reason = ""
        
        if n_cells == 0:
            phase = "DORMANT"
            phase_2_reason = "No methodology cells yet. Start interacting with IA."
        elif n_cells < 50:
            phase = "SEEDING"
            phase_2_reason = f"Need 50+ methodology cells. Currently: {n_cells}. Keep interacting."
        elif avg_connections < 2.0:
            phase = "GERMINATING"
            phase_2_reason = f"Cells need more interactions. Avg connections: {avg_connections:.1f} (need 2.0+). Keep interacting."
        elif n_cells >= 50 and avg_connections >= 2.0 and len(self._assimilation_log) == 0:
            phase = "READY_FOR_PHASE_2"
            phase_2_ready = True
            phase_2_reason = "✅ Sufficient methodology cells and connection density. Phase 2 can begin."
        elif len(self._assimilation_log) > 0:
            avg_sim = sum(self._assimilation_log[-50:]) / len(self._assimilation_log[-50:])
            phase_2_ready = True
            if avg_sim < 0.30:
                phase = "ABSORBING"
                phase_2_reason = f"Phase 2 active. Avg similarity: {avg_sim:.1%} — early learning, high error."
            elif avg_sim < 0.70:
                phase = "CONVERGING"
                phase_2_reason = f"Phase 2 active. Avg similarity: {avg_sim:.1%} — patterns forming."
            elif avg_sim < 0.90:
                phase = "MIRRORING"
                phase_2_reason = f"Phase 2 active. Avg similarity: {avg_sim:.1%} — closely tracking base model."
            else:
                phase = "READY"
                phase_2_reason = f"Phase 2 active. Avg similarity: {avg_sim:.1%} — 🟢 INDEPENDENCE THRESHOLD REACHED."
        
        avg_similarity = 0.0
        if self._assimilation_log:
            recent = self._assimilation_log[-50:]
            avg_similarity = sum(recent) / len(recent)
        
        return {
            "methodology_cells": n_cells,
            "total_connections": total_connections,
            "avg_connections": round(avg_connections, 2),
            "avg_activations": round(avg_activations, 1),
            "avg_energy": round(avg_energy, 1),
            "interactions_total": self._methodology_interactions,
            "phase": phase,
            "phase_2_ready": phase_2_ready,
            "phase_2_status": phase_2_reason,
            "avg_similarity": round(avg_similarity, 4),
            "similarity_samples": len(self._assimilation_log),
        }

    def _rebuild_lobe_index(self):
        """Build per-mode cell caches from current methodology cells."""
        self._lobe_index = {}
        all_cells = list(self.methodology_cells.values())
        
        for mode, config in self.LOBE_CONFIG.items():
            dominant_prefixes = config["dominant"]
            tools_prefixes = config["tools"]
            
            dominant_cells = []
            tools_cells = []
            
            for cell in all_cells:
                cid = cell.cell_id
                is_dominant = any(cid.startswith(p) for p in dominant_prefixes)
                is_tool = any(cid.startswith(p) for p in tools_prefixes)
                
                if is_dominant:
                    dominant_cells.append(cell)
                elif is_tool:
                    tools_cells.append(cell)
            
            self._lobe_index[mode] = {
                "dominant": dominant_cells,
                "tools": tools_cells,
            }
        
        self._lobe_dirty = False
        
        # Log the index distribution
        for mode, idx in self._lobe_index.items():
            d_count = len(idx["dominant"])
            t_count = len(idx["tools"])
            print(f"[Lobe Index] {mode}: {d_count:,} dominant + {t_count:,} tools = {d_count + t_count:,} cells")
    
    def predict(self, query_text: str, mode: str = "speech",
                exclude_cells: t.Optional[t.Set[str]] = None) -> dict:
        """
        [PHASE 3: LOBE-ROUTED PREDICTION]
        
        Routes queries to the appropriate brain lobe based on intent.
        Conversation cells are the DOMINANT frame; dictionary cells are TOOLS.
        
        Priority weighting:
          - Dominant cells: similarity × dominant_weight (e.g. 3x for speech cells)
          - Tool cells: similarity × tools_weight (e.g. 1x for dictionary cells)
        
        This means a speech cell at 0.20 similarity scores 0.60 (dominant),
        while a dictionary cell at 0.25 similarity scores only 0.25 (tool).
        
        exclude_cells: Set of cell IDs to hard-exclude (e.g. from negative feedback)
        """
        ACTIVATION_THRESHOLD = 0.05
        
        # Rebuild lobe index if needed
        if self._lobe_dirty:
            self._rebuild_lobe_index()
        
        # Get the lobe config
        config = self.LOBE_CONFIG.get(mode, self.LOBE_CONFIG["speech"])
        lobe_data = self._lobe_index.get(mode, self._lobe_index["speech"])
        
        dominant_weight = config["dominant_weight"]
        tools_weight = config["tools_weight"]
        
        # Combine dominant + tools cells into one pool with weight tags
        cells = []
        cell_weights = []
        
        for cell in lobe_data["dominant"]:
            if exclude_cells and cell.cell_id in exclude_cells:
                continue
            cells.append(cell)
            cell_weights.append(dominant_weight)
        
        for cell in lobe_data["tools"]:
            if exclude_cells and cell.cell_id in exclude_cells:
                continue
            cells.append(cell)
            cell_weights.append(tools_weight)
        
        if len(cells) < 2:
            return {"prediction": None, "activated_cells": [], "error": f"Not enough cells in '{mode}' lobe"}
        
        weight_array = np.array(cell_weights, dtype=np.float32)
        
        # Encode the query
        query_vec = self.hdc.encode(query_text)
        query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-8)
        
        # VECTORIZED: Score all cells in this lobe
        dna_matrix = np.array([c.dna for c in cells], dtype=np.float32)
        raw_similarities = dna_matrix @ query_vec
        
        # Apply lobe priority weighting: dominant cells get boosted
        weighted_similarities = raw_similarities * weight_array
        
        # All cells above threshold (WIDE READ) — using weighted scores
        above_mask = weighted_similarities > ACTIVATION_THRESHOLD
        activated_indices = np.where(above_mask)[0]
        
        if len(activated_indices) == 0:
            best_sim = float(np.max(weighted_similarities))
            return {
                "prediction": None,
                "activated_cells": [],
                "error": f"No cells above threshold in '{mode}' lobe. Best: {best_sim:.4f}"
            }
        
        # Sort by weighted similarity descending
        sorted_activated = activated_indices[np.argsort(weighted_similarities[activated_indices])[::-1]]
        
        # TIERED: Precision Learning — Only top 5 to 10 cells get reinforced
        # Avoids reinforcing generic fluff cells in the long tail
        learn_n = max(5, min(10, int(np.sqrt(len(sorted_activated)))))
        learn_indices = sorted_activated[:learn_n]
        
        # Activate the LEARNING cells — weighted by confidence
        outputs = []
        weights = []
        learn_ids = []
        learn_cells = []
        for idx in learn_indices:
            cell = cells[idx]
            output_vec = cell.activate(query_vec)
            conf = getattr(cell, 'confidence', 1.0)
            outputs.append(output_vec * conf)
            weights.append(conf)
            learn_ids.append(cell.cell_id)
            learn_cells.append(cell)
        
        # Confidence-weighted average of outputs
        total_weight = sum(weights)
        if total_weight > 0:
            prediction = sum(outputs) / total_weight
        else:
            prediction = np.mean(outputs, axis=0)
        prediction = prediction / max(np.linalg.norm(prediction), 1e-8)
        
        # ── Spreading activation — expand pool via connection graph ──────────────
        # For each top-matched cell, follow its strongest connections to pull in
        # semantically linked cells that cosine similarity alone wouldn't surface.
        # This is the bridge between lookup and pattern-based understanding.
        cell_id_set = {c.cell_id for c in learn_cells}
        connection_candidates = {}  # cell_id → activation_score

        for cell in learn_cells[:5]:  # only expand from top 5 direct matches
            for connected_id, link_strength in sorted(
                cell.connections.items(), key=lambda x: -x[1]
            )[:4]:  # top 4 connections per cell
                if connected_id in cell_id_set:
                    continue
                connected_cell = self.methodology_cells.get(connected_id)
                if connected_cell is None:
                    continue
                score = link_strength * getattr(connected_cell, 'confidence', 1.0) * 0.6
                if connected_id not in connection_candidates or connection_candidates[connected_id] < score:
                    connection_candidates[connected_id] = score

        for cid, _ in sorted(connection_candidates.items(), key=lambda x: -x[1])[:5]:
            connected_cell = self.methodology_cells[cid]
            learn_cells.append(connected_cell)
            learn_ids.append(cid)
            cell_id_set.add(cid)
        # ─────────────────────────────────────────────────────────────────────────

        # Build proposal text from top 10 activated cells (for display)
        proposals = []
        for idx in sorted_activated[:10]:
            cell = cells[idx]
            sim = float(weighted_similarities[idx])
            raw_sim = float(raw_similarities[idx])
            conf = getattr(cell, 'confidence', 1.0)
            src = getattr(cell, 'source', 'dict')[:6]
            proposals.append(f"({raw_sim:.2f}|{src}) {cell.content}")
        
        prediction_text = "Neural Solution Found:\n" + "\n".join(proposals)
        
        active_lobe = mode
        
        return {
            "prediction": prediction.tolist(),
            "activated_cells": learn_cells,
            "activated_cell_ids": learn_ids,
            "prediction_text": prediction_text,
            "active_lobe": active_lobe,
            "total_activated": len(activated_indices),
            "learn_cells": len(learn_ids),
            "lobe_pool_size": len(cells),
        }
    
    def learn_from_signal(self, cell_ids: list, target_vector: list, similarity: float):
        """
        [PHASE 2: STUDENT CORRECTION — RANK-WEIGHTED]
        
        Top cells (highest similarity) learn more aggressively.
        Bottom cells learn less. This focuses the learning signal
        on the most relevant cells and prevents low-similarity
        cells from drifting their DNA toward unrelated signals.
        """
        target = np.array(target_vector, dtype=np.float32)
        target_norm = np.linalg.norm(target)
        
        # Guard: refuse to learn from a zero/near-zero target vector
        if target_norm < 1e-6:
            print(f"[Assimilation] SKIPPED: Zero target vector received. No cell weights adjusted.")
            return {"learned": 0, "similarity": similarity, "error": "zero_target_vector"}
        
        target = target / target_norm
        
        learned_count = 0
        n_cells = len(cell_ids)
        for i, cell_id in enumerate(cell_ids):
            cell = self.methodology_cells.get(cell_id)
            if cell:
                # Rank-weighted learning: top cell gets 1.0x, bottom gets 0.3x
                rank_scale = 1.0 - (i / max(n_cells, 1)) * 0.7
                cell.learn(target, learning_rate=0.001 * rank_scale)
                learned_count += 1
        
        self._assimilation_log.append(float(similarity))
        
        if len(self._assimilation_log) > 500:
            self._assimilation_log = self._assimilation_log[-500:]
        
        avg_recent = sum(self._assimilation_log[-20:]) / max(len(self._assimilation_log[-20:]), 1)
        print(f"[Assimilation] Phase 2 learning: {learned_count} cells updated | Similarity: {similarity:.3f} | Avg: {avg_recent:.3f}")
        
        return {"learned": learned_count, "similarity": similarity}
    
    # ──────────────────────────────────────────────────
    # LIVE CONVERSATIONAL INTELLIGENCE METHODS
    # ──────────────────────────────────────────────────
    
    def suppress_cells(self, cell_ids: list, factor: float = 0.1) -> dict:
        """
        Slash the confidence of cells that produced a bad response.
        factor=0.1 means a cell at confidence 1.0 drops to 0.1.
        Also drains energy so the cell becomes dormant faster.
        """
        suppressed = 0
        for cell_id in cell_ids:
            cell = self.methodology_cells.get(cell_id)
            if cell:
                old_conf = cell.confidence
                cell.confidence = max(0.01, cell.confidence * factor)
                cell.energy = max(0.0, cell.energy - 50.0)
                suppressed += 1
                self._persist_methodology_cell(cell)
        
        if suppressed > 0:
            self._lobe_dirty = True  # Confidence changed, may affect behavior
        
        return {"suppressed": suppressed, "factor": factor}
    
    def reinforce_cells(self, cell_ids: list, boost: float = 0.02,
                        query_vec: np.ndarray = None, drift: float = 0.02) -> dict:
        """
        Boost the confidence of cells that produced an accepted response.
        If query_vec is provided, also drift each cell's DNA toward the query —
        so cells self-organise into the semantic territory where they actually work.
        """
        reinforced = 0
        for cell_id in cell_ids:
            cell = self.methodology_cells.get(cell_id)
            if cell:
                cell.confidence = min(1.5, cell.confidence + boost)
                cell.energy = min(100.0, cell.energy + 2.0)
                if query_vec is not None and drift > 0:
                    cell.learn(query_vec, learning_rate=drift)
                reinforced += 1

        return {"reinforced": reinforced, "boost": boost}
    
    def anti_learn(self, cell_ids: list, query_vector: list):
        """
        Push cell DNA vectors AWAY from the query vector.
        This is the opposite of learn() — it creates negative associations
        so these cells are less likely to fire for similar queries in the future.
        """
        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-6:
            return
        query = query / query_norm
        
        for cell_id in cell_ids:
            cell = self.methodology_cells.get(cell_id)
            if cell:
                # Push DNA away from the query direction
                # Subtract a fraction of the query from the cell's DNA
                cell.dna = cell.dna - query * 0.01
                norm = np.linalg.norm(cell.dna)
                if norm > 0:
                    cell.dna = cell.dna / norm
                self._persist_methodology_cell(cell)
    
    def ingest_creator_cell(self, query_text: str, correct_answer: str) -> str:
        """
        Ingest a creator-taught cell at the HIGHEST confidence (1.5).
        The creator's word is law. These cells outweigh everything else.
        """
        content_hash = hashlib.md5(f"{query_text}{correct_answer}".encode()).hexdigest()[:12]
        cell_id = f"meth_creator_{content_hash}"
        
        if cell_id in self.methodology_cells:
            # Update existing creator cell — boost it even further
            existing = self.methodology_cells[cell_id]
            existing.confidence = min(1.5, existing.confidence + 0.1)
            existing.energy = 100.0
            self._persist_methodology_cell(existing)
            return cell_id
        
        combined_text = f"{query_text} {correct_answer}"
        dna_vector = self.hdc.encode(combined_text)
        dna_norm = np.linalg.norm(dna_vector)
        if dna_norm < 1e-8:
            return None
        
        cell = LivingCell(
            cell_id,
            content=correct_answer,  # <--- CRITICAL FIX: The payload is ONLY the answer
            source_table="creator_teaching",
            confidence=1.5,
            source="creator"
        )
        cell.dna = dna_vector / dna_norm
        
        self.methodology_cells[cell_id] = cell
        self._persist_methodology_cell(cell)
        self._lobe_dirty = True  # New cell — rebuild index
        
        print(f"[Creator] New teaching cell: {cell_id} | confidence: 1.5")
        return cell_id
    
    def ingest_conversation_cell(self, query_text: str, response_text: str,
                                  confidence: float = 0.8, source: str = "conversation") -> str:
        """
        Birth a new cell from a live conversation exchange.
        Called when the user implicitly accepts a response (doesn't correct it).
        """
        content_hash = hashlib.md5(f"{query_text}{response_text}".encode()).hexdigest()[:12]
        cell_id = f"meth_conv_{content_hash}"
        
        if cell_id in self.methodology_cells:
            return None  # Already exists
        
        combined_text = f"{query_text} {response_text}"
        dna_vector = self.hdc.encode(combined_text)
        dna_norm = np.linalg.norm(dna_vector)
        if dna_norm < 1e-8:
            return None
        
        cell = LivingCell(
            cell_id,
            content=combined_text,
            source_table="live_conversation",
            confidence=confidence,
            source=source
        )
        cell.dna = dna_vector / dna_norm
        
        self.methodology_cells[cell_id] = cell
        self._persist_methodology_cell(cell)
        self._lobe_dirty = True
        
        return cell_id

    def get_lobe_content_sample(self, mode: str, limit: int = 20) -> t.List[str]:
        """
        Sample content from a specific brain lobe.
        This gives the Teacher a 'mirror' into what DataG already knows,
        so it can identify gaps and generate targeted training data.
        """
        import random
        
        if self._lobe_dirty:
            self._rebuild_lobe_index()
        
        lobe_data = self._lobe_index.get(mode)
        if not lobe_data:
            return []
        
        # Sample from the dominant cells (these represent the lobe's core knowledge)
        dominant = lobe_data.get("dominant", [])
        if not dominant:
            return []
        
        sample_size = min(limit, len(dominant))
        sampled = random.sample(dominant, sample_size)
        
        return [cell.content[:200] for cell in sampled]
    
    def get_lobe_stats(self) -> t.Dict[str, dict]:
        """
        Get statistics for all brain lobes — cell counts, avg confidence, avg energy.
        Useful for identifying which lobes need growth.
        """
        if self._lobe_dirty:
            self._rebuild_lobe_index()
        
        stats = {}
        for mode, lobe_data in self._lobe_index.items():
            dominant = lobe_data.get("dominant", [])
            tools = lobe_data.get("tools", [])
            
            if dominant:
                avg_conf = sum(c.confidence for c in dominant) / len(dominant)
                avg_energy = sum(c.energy for c in dominant) / len(dominant)
            else:
                avg_conf = 0.0
                avg_energy = 0.0
            
            stats[mode] = {
                "dominant_count": len(dominant),
                "tools_count": len(tools),
                "total": len(dominant) + len(tools),
                "avg_confidence": round(avg_conf, 3),
                "avg_energy": round(avg_energy, 1),
            }
        
        return stats

    def get_telemetry(self) -> t.Dict[str, t.Any]:
        return {cid: cell.perceive()["state"] for cid, cell in self.cells.items()}
