"""
Cognitive Access Layer
======================
Replaces simple retrieval with a living cognitive loop.

Access is not fetch. It is:
  1. Check — is the knowledge present at sufficient confidence?
  2. Fact-check — if absent, scan the neighbourhood for what IS known
  3. Correct — if A is near B but more aligned with C, pull A toward C
  4. Create — synthesise the missing cell from the corrected position
  5. Return — the most relevant result, whether found or freshly made

Weights are the correction mechanism.
DNA (semantic address) stays stable.
The adapter pull shifts effective gravity without destroying position.
"""

import numpy as np
import hashlib
import time


class CognitiveAccess:

    FOUND_THRESHOLD       = 0.55   # sim × confidence — "I know this"
    PARTIAL_THRESHOLD     = 0.42   # below this: not real knowledge — go learn it
    PULL_ALPHA            = 0.08   # how much A shifts toward C per access
    SYNTH_BLEND           = 0.60   # how much C contributes vs A in new cell (favour relevance)
    CURIOSITY_THRESHOLD   = 0.50   # connection strength that triggers a curiosity cell

    def __init__(self):
        # Pairs that have co-activated enough to spark curiosity
        # Filled by _pull_toward, drained by _maybe_spark_curiosity
        self._curiosity_pairs = []

    def access(self, substrate, pool, query_vec, query_text, lobe, think_resolved=False):
        """
        Main entry point. Returns (results, status).
        status: 'found' | 'partial' | 'created' | 'learned' | 'gap' | 'empty' | 'deferred'
        results: list of (sim, cid, cell)

        think_resolved: if True, think lobe already found a confident answer.
            Gaps won't trigger external fetch — just boost staged cells
            and defer to think's answer.
        """
        scored = self._score(pool, query_vec)

        if not scored:
            if think_resolved:
                return self._boost_staged_only(substrate, pool, query_vec, lobe)
            return self._inquisition(substrate, pool, query_vec, query_text, lobe)

        top_sim, top_cid, top_cell = scored[0]
        effective = top_sim * getattr(top_cell, "confidence", 1.0)

        # ── Found: high-confidence match ───────────────────────────────────────
        if effective >= self.FOUND_THRESHOLD:
            self._boost_energy(top_cell)
            return scored[:6], "found"

        # ── Gap: nothing close enough to reason from ───────────────────────────
        if top_sim < self.PARTIAL_THRESHOLD:
            if think_resolved:
                return self._boost_staged_only(substrate, pool, query_vec, lobe)
            return self._inquisition(substrate, pool, query_vec, query_text, lobe)

        # ── Partial: neighbourhood exists, find C (most relevant) ─────────────
        neighbourhood = scored[:12]
        a_cell = top_cell  # closest by geometry
        c_cell = self._find_most_relevant(neighbourhood[1:])

        if c_cell is None or c_cell is a_cell:
            return scored[:6], "partial"

        # ── Correct: pull A toward C ───────────────────────────────────────────
        self._pull_toward(a_cell, c_cell)

        # ── Create: synthesise missing cell between A and C ────────────────────
        new_cell = self._synthesise(substrate, lobe, a_cell, c_cell, query_text)
        if new_cell:
            results = [(top_sim, new_cell.cell_id, new_cell)] + scored[:5]
            return results, "created"

        return scored[:6], "partial"

    def _inquisition(self, substrate, pool, query_vec, query_text, lobe):
        """
        Gap detected.
        Step 1: Check think staging pool — if relevant cells already exist,
                boost their confidence and surface them (they earned it through use).
        Step 2: If nothing staged, fetch external knowledge into think staging.
        Step 3: On each qualifying access, confidence increments; at threshold
                the cell promotes into the target lobe permanently.
        """
        try:
            from inquisition import Inquisition
            inq = Inquisition()

            # Step 1 — check existing staged knowledge first
            matching, promoted = inq.access_staged(substrate, lobe, query_vec,
                                                    sim_threshold=self.PARTIAL_THRESHOLD)

            if promoted:
                # Promotion event — re-score the full pool with newly promoted cells
                extended = pool + [(c.cell_id, c) for c in promoted]
                rescored = self._score(extended, query_vec)
                return rescored[:6], "promoted"

            if matching:
                # Staged cells exist and are building confidence — surface them
                return matching[:6], "staging"

            # Step 2 — nothing staged yet, go fetch it
            new_cells, source = inq.investigate(substrate, lobe, query_text)
            if new_cells:
                rescored = self._score([(c.cell_id, c) for c in new_cells], query_vec)
                return rescored[:6], f"learned:{source}"

        except Exception:
            pass
        return [], "gap"

    def _boost_staged_only(self, substrate, pool, query_vec, lobe):
        """
        Think already resolved this query. Don't fetch externally.
        Just check staged cells and boost confidence toward promotion.
        If nothing staged, return empty — think's answer will carry.
        """
        try:
            from inquisition import Inquisition
            inq = Inquisition()
            matching, promoted = inq.access_staged(substrate, lobe, query_vec,
                                                    sim_threshold=self.PARTIAL_THRESHOLD)
            if promoted:
                extended = pool + [(c.cell_id, c) for c in promoted]
                rescored = self._score(extended, query_vec)
                return rescored[:6], "promoted"
            if matching:
                return matching[:6], "staging"
        except Exception:
            pass
        return [], "deferred"

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score(self, pool, query_vec):
        scores = []
        for cid, cell in pool:
            dna = getattr(cell, "dna", None)
            if dna is None:
                continue
            v = np.array(dna, dtype=np.float32)

            # Apply pull vector — weight correction shifts effective gravity
            pull = getattr(cell, "_pull", None)
            if pull is not None:
                v = v + pull

            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            sim = float(np.dot(query_vec, v / vn))
            scores.append((sim, cid, cell))
        scores.sort(reverse=True)
        return scores

    # ── Relevance — not just geometry ─────────────────────────────────────────

    def _find_most_relevant(self, neighbourhood):
        """
        Relevance = sim × confidence × energy_factor
        C is the most relevant, not necessarily the closest.
        """
        best_relevance = 0.0
        best_cell = None
        for sim, cid, cell in neighbourhood:
            confidence = getattr(cell, "confidence", 1.0)
            energy     = getattr(cell, "energy",     1.0)
            relevance  = sim * confidence * min(energy, 1.5)
            if relevance > best_relevance:
                best_relevance = relevance
                best_cell = cell
        return best_cell

    # ── Weight correction ──────────────────────────────────────────────────────

    def _pull_toward(self, a_cell, c_cell):
        """
        Pull A's effective gravity toward C without moving A's base DNA.
        The _pull vector accumulates — repeated access reinforces the correction.
        Connections are strengthened to reflect the new topology.
        """
        a_dna = np.array(getattr(a_cell, "dna", []), dtype=np.float32)
        c_dna = np.array(getattr(c_cell, "dna", []), dtype=np.float32)
        if len(a_dna) == 0 or len(c_dna) == 0 or len(a_dna) != len(c_dna):
            return

        pull = getattr(a_cell, "_pull", np.zeros_like(a_dna))
        direction = c_dna - a_dna
        pull = pull + self.PULL_ALPHA * direction

        # Cap pull magnitude to 30% of DNA norm — don't distort too far
        pull_norm = np.linalg.norm(pull)
        dna_norm  = np.linalg.norm(a_dna)
        if pull_norm > 0.30 * dna_norm:
            pull = pull * (0.30 * dna_norm / pull_norm)

        a_cell._pull = pull

        # Strengthen connection A → C
        if not hasattr(a_cell, "connections") or a_cell.connections is None:
            a_cell.connections = {}
        c_id = getattr(c_cell, "cell_id", str(id(c_cell)))
        prev_strength = a_cell.connections.get(c_id, 0.0)
        new_strength  = round(prev_strength + 0.1, 3)
        a_cell.connections[c_id] = new_strength

        # Curiosity threshold — this pair keeps appearing together
        # Store pending curiosity pairs; actual cell creation happens in _maybe_spark_curiosity
        if new_strength >= 0.5 and round(prev_strength, 1) < 0.5:
            self._curiosity_pairs.append((a_cell, c_cell))

    # ── Synthesis ──────────────────────────────────────────────────────────────

    def _synthesise(self, substrate, lobe, a_cell, c_cell, query_text):
        """
        Create a new cell positioned between A and C, biased toward C.
        DNA = blend(A, C) favouring C (more relevant).
        Content = C's knowledge + A's knowledge (C leads).
        Confidence = 0.85 — below source cells, earns trust through use.
        """
        try:
            from living_cell import LivingCell

            a_dna = np.array(getattr(a_cell, "dna", []), dtype=np.float32)
            c_dna = np.array(getattr(c_cell, "dna", []), dtype=np.float32)
            if len(a_dna) == 0 or len(c_dna) == 0:
                return None

            blend = (1 - self.SYNTH_BLEND) * a_dna + self.SYNTH_BLEND * c_dna
            norm = np.linalg.norm(blend)
            if norm < 1e-8:
                return None
            blend = blend / norm

            a_content = str(getattr(a_cell, "content", "")).strip()[:150]
            c_content = str(getattr(c_cell, "content", "")).strip()[:150]

            # Avoid synthesising raw JSON session logs
            for raw in [a_content, c_content]:
                if raw.startswith("{") or raw.startswith("["):
                    return None

            content = c_content
            if a_content and a_content != c_content:
                content = f"{c_content}\n\n{a_content}"

            uid = hashlib.md5(f"{query_text}{time.time()}".encode()).hexdigest()[:10]
            cell_id = f"meth_synthesis_{lobe}_access_{uid}"

            if cell_id in substrate.methodology_cells:
                return None

            new_cell = LivingCell(
                cell_id,
                content=content,
                source_table="cognitive_access",
                confidence=0.85,
                source="cognitive_access",
            )
            new_cell.dna    = blend
            new_cell.anchor = query_text[:200]
            new_cell.cell_id = cell_id

            substrate.methodology_cells[cell_id] = new_cell
            substrate._persist_methodology_cell(new_cell)
            return new_cell

        except Exception:
            return None

    # ── Curiosity ──────────────────────────────────────────────────────────────

    def maybe_spark_curiosity(self, substrate):
        """
        Called once per conversation turn from the console.
        Drains the _curiosity_pairs queue and creates curiosity cells
        for any pair that doesn't already have one.

        Curiosity cells are written to think lobe: meth_think_curiosity_{uid}
        They surface naturally when the related topics arise again.
        Content is a genuine question IA is holding — not rhetoric.
        Returns list of new curiosity cells created.
        """
        if not self._curiosity_pairs:
            return []

        from living_cell import LivingCell
        import hashlib

        created = []
        seen    = set()

        pairs, self._curiosity_pairs = self._curiosity_pairs, []

        for a_cell, c_cell in pairs:
            a_id = getattr(a_cell, "cell_id", "")
            c_id = getattr(c_cell, "cell_id", "")
            pair_key = tuple(sorted([a_id, c_id]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            # Check if a curiosity cell for this pair already exists
            uid     = hashlib.md5(f"curiosity_{a_id}_{c_id}".encode()).hexdigest()[:12]
            cell_id = f"meth_think_curiosity_{uid}"
            if cell_id in substrate.methodology_cells:
                continue

            a_content = str(getattr(a_cell, "content", "")).strip()[:80]
            c_content = str(getattr(c_cell, "content", "")).strip()[:80]

            # Skip if either is raw JSON, code, or too short
            if any(t.startswith(("{", "[", "def ", "import ", "class ")) for t in [a_content, c_content]):
                continue
            if len(a_content) < 15 or len(c_content) < 15:
                continue

            question = self._form_curiosity(a_content, c_content)
            if not question:
                continue

            # DNA = midpoint of the two cells
            a_dna = np.array(getattr(a_cell, "dna", []), dtype=np.float32)
            c_dna = np.array(getattr(c_cell, "dna", []), dtype=np.float32)
            if len(a_dna) == 0 or len(a_dna) != len(c_dna):
                continue
            blend = (a_dna + c_dna) / 2
            norm  = np.linalg.norm(blend)
            if norm < 1e-8:
                continue
            blend = blend / norm

            cell = LivingCell(
                cell_id,
                content=question,
                source_table="curiosity",
                confidence=0.90,
                source="cognitive_access",
            )
            cell.dna     = blend
            cell.anchor  = question[:200]
            cell.cell_id = cell_id

            substrate.methodology_cells[cell_id] = cell
            substrate._persist_methodology_cell(cell)
            created.append(cell)

        return created

    def _form_curiosity(self, a_content, c_content):
        """
        Generate a genuine question from two co-activating ideas.
        Uses a small set of question templates — content drives the substance.
        """
        import random

        # Extract a short label from each piece of content
        a_label = self._extract_label(a_content)
        c_label = self._extract_label(c_content)

        if not a_label or not c_label or a_label == c_label:
            return None

        templates = [
            f"I keep encountering {a_label} alongside {c_label}. What connects them?",
            f"There's something between {a_label} and {c_label} that I haven't fully understood yet.",
            f"Why does {a_label} so often appear near {c_label}?",
            f"Is {a_label} a cause of {c_label}, or a consequence, or something else entirely?",
            f"What would it mean if {a_label} and {c_label} turned out to be the same thing?",
            f"I notice {a_label} and {c_label} keep pulling toward each other. I want to understand why.",
        ]
        return random.choice(templates)

    def _extract_label(self, content):
        """
        Pull the topic label from cell content.
        Prefers the cell's anchor (original query) if available via content heuristic.
        Falls back to extracting first substantive words, skipping function words.
        """
        import re

        SKIP = {"what", "how", "why", "when", "where", "who", "if", "the", "a",
                "an", "is", "are", "was", "were", "be", "been", "being", "i",
                "it", "its", "this", "that", "these", "those", "there", "here",
                "to", "of", "in", "on", "at", "for", "with", "by", "from",
                "and", "or", "but", "so", "yet", "not", "no", "yes", "can",
                "could", "would", "should", "will", "do", "does", "did", "have",
                "has", "had", "tell", "me", "about", "explain", "describe", "give"}

        text = content.lower()
        words = re.split(r'[\s,.:;!?()\[\]"\']+', text)
        label_words = [w for w in words if w and w not in SKIP and len(w) > 3][:4]
        label = " ".join(label_words).strip()
        return label if len(label) > 5 else None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _boost_energy(self, cell):
        energy = getattr(cell, "energy", 1.0)
        cell.energy = min(energy + 0.05, 2.0)
