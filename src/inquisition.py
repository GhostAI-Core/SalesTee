"""
Inquisition — Gap-Driven Knowledge Acquisition
===============================================
When cognitive access hits a gap, IA doesn't say no.
It fetches the answer, stages it in think, and lets it earn
its way into the initiating lobe through repeated access.

The coffee shop that learns tea — then masters it.
At scale: every unknown is a growth event.

Staging model:
  1. External knowledge ALWAYS lands in think lobe first.
     cell_id: meth_think_inquisition_{uid}   confidence=0.75
  2. Each access from any lobe increments confidence (+0.10)
  3. When confidence >= PROMOTION_THRESHOLD (1.2), the cell
     is promoted: a copy is written to the initiating lobe
     with confidence=1.0 and _promoted=True flagged on the
     think staging cell.
  4. Think cell remains as the thought record — it still
     reasoned about it.

Sources (in priority order):
  1. Wikipedia — encyclopedic, reliable, vast
  2. Wiktionary — definitions, etymology, language
"""

import urllib.request
import urllib.parse
import json
import hashlib
import numpy as np


class Inquisition:

    MAX_CHUNKS          = 12    # max cells to create per investigation
    MIN_LENGTH          = 60    # minimum character length for a chunk to be useful
    CONFIDENCE          = 0.75  # staged knowledge starts unverified
    PROMOTION_THRESHOLD = 1.2   # confidence level at which think cell promotes to target lobe
    CONFIDENCE_INCREMENT = 0.10 # confidence boost per qualifying access
    TIMEOUT             = 6     # seconds per source request

    SOURCES = [
        "wikipedia",
        "wiktionary",
    ]

    # ── Public interface ───────────────────────────────────────────────────────

    def investigate(self, substrate, target_lobe, query_text):
        """
        Fetch knowledge about query_text from external sources.
        ALL knowledge stages into think lobe first (meth_think_inquisition_*).
        Returns (staged_cells, source_name) — or ([], None) if nothing found.
        target_lobe is recorded on each cell for later promotion.
        """
        candidates = self._query_candidates(query_text)
        for source in self.SOURCES:
            for candidate in candidates:
                chunks, source_name = self._fetch(source, candidate)
                if chunks:
                    new_cells = self._ingest(substrate, target_lobe, chunks, query_text, source_name)
                    if new_cells:
                        return new_cells, source_name

        return [], None

    def access_staged(self, substrate, target_lobe, query_vec, sim_threshold=0.42):
        """
        Scan think staging pool for cells relevant to this query.
        Increment confidence on qualifying matches.
        Promote any cell that hits PROMOTION_THRESHOLD.
        Returns (matching_cells, promoted_cells).
        """
        matching   = []
        promoted   = []

        think_pool = [
            (cid, cell)
            for cid, cell in substrate.methodology_cells.items()
            if cid.startswith("meth_think_inquisition_")
               and not cell.meta.get("promoted", False)
        ]

        for cid, cell in think_pool:
            dna = getattr(cell, "dna", None)
            if dna is None:
                continue
            v  = np.array(dna, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            sim = float(np.dot(query_vec, v / vn))
            if sim < sim_threshold:
                continue

            # Qualifying hit — increment confidence
            cell.confidence = round(getattr(cell, "confidence", self.CONFIDENCE) + self.CONFIDENCE_INCREMENT, 3)
            matching.append((sim, cid, cell))

            # Check for promotion
            if cell.confidence >= self.PROMOTION_THRESHOLD:
                promoted_cell = self._promote(substrate, cell, target_lobe)
                if promoted_cell:
                    promoted.append(promoted_cell)

        # Persist updated confidence values
        for _, cid, cell in matching:
            try:
                substrate._persist_methodology_cell(cell)
            except Exception:
                pass

        matching.sort(reverse=True)
        return matching, promoted

    def _promote(self, substrate, think_cell, target_lobe):
        """
        Graduate a think-staged cell into its target lobe.
        Creates meth_{target_lobe}_inquisition_{uid} with confidence=1.0.
        Flags think cell as _promoted so it doesn't promote again.
        """
        from living_cell import LivingCell

        uid      = getattr(think_cell, "cell_id", "").split("_")[-1]
        cell_id  = f"meth_{target_lobe}_inquisition_{uid}"

        if cell_id in substrate.methodology_cells:
            think_cell.meta["promoted"] = True
            return None

        dna = getattr(think_cell, "dna", None)
        if dna is None:
            return None

        promoted = LivingCell(
            cell_id,
            content=getattr(think_cell, "content", ""),
            source_table=getattr(think_cell, "source_table", "inquisition"),
            confidence=1.0,
            source="inquisition_promoted",
        )
        promoted.dna      = np.array(dna, dtype=np.float32)
        promoted.anchor   = getattr(think_cell, "anchor", "")
        promoted.cell_id  = cell_id

        substrate.methodology_cells[cell_id] = promoted
        substrate._persist_methodology_cell(promoted)

        think_cell.meta["promoted"] = True
        substrate._persist_methodology_cell(think_cell)   # persist promoted flag
        return promoted

    def _query_candidates(self, query_text):
        """
        Generate progressively simpler search terms from the raw query.
        Strips question words and filler so Wikipedia gets a clean title.
        """
        STRIP_PREFIXES = [
            "tell me about ", "what is ", "what are ", "who is ", "who was ",
            "explain ", "describe ", "how does ", "how do ", "why is ", "why does ",
            "can you explain ", "can you tell me about ", "what do you know about ",
            "i want to know about ", "give me information on ", "give me info on ",
        ]
        q = query_text.strip().rstrip("?.")
        candidates = [q]

        lower = q.lower()
        for prefix in STRIP_PREFIXES:
            if lower.startswith(prefix):
                stripped = q[len(prefix):].strip()
                if stripped and stripped not in candidates:
                    candidates.append(stripped)
                break

        # Also try last 2-4 meaningful words if query is long
        words = q.split()
        if len(words) > 4:
            candidates.append(" ".join(words[-3:]))
            candidates.append(" ".join(words[-4:]))

        return candidates

    # ── Fetch ──────────────────────────────────────────────────────────────────

    def _fetch(self, source, query):
        if source == "wikipedia":
            return self._fetch_wikipedia(query)
        if source == "wiktionary":
            return self._fetch_wiktionary(query)
        return [], source

    def _fetch_wikipedia(self, query):
        try:
            term  = urllib.parse.quote(query.replace(" ", "_"))
            url   = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query&format=json&prop=extracts"
                f"&exintro=0&explaintext=1&titles={term}&redirects=1"
            )
            req  = urllib.request.Request(url, headers={"User-Agent": "IA-Inquisition/1.0"})
            resp = urllib.request.urlopen(req, timeout=self.TIMEOUT)
            data = json.loads(resp.read())
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if page.get("pageid", -1) == -1:
                    continue  # not found
                extract = page.get("extract", "")
                if len(extract) < self.MIN_LENGTH:
                    continue
                chunks = self._chunk_text(extract)
                return chunks, "wikipedia"
        except Exception:
            pass
        return [], "wikipedia"

    def _fetch_wiktionary(self, query):
        try:
            term  = urllib.parse.quote(query.strip().lower())
            url   = (
                f"https://en.wiktionary.org/w/api.php"
                f"?action=query&format=json&prop=extracts"
                f"&explaintext=1&titles={term}"
            )
            req  = urllib.request.Request(url, headers={"User-Agent": "IA-Inquisition/1.0"})
            resp = urllib.request.urlopen(req, timeout=self.TIMEOUT)
            data = json.loads(resp.read())
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if page.get("pageid", -1) == -1:
                    continue
                extract = page.get("extract", "")
                if len(extract) < self.MIN_LENGTH:
                    continue
                chunks = self._chunk_text(extract)
                return chunks, "wiktionary"
        except Exception:
            pass
        return [], "wiktionary"

    # ── Chunk ──────────────────────────────────────────────────────────────────

    def _chunk_text(self, text):
        """
        Split into meaningful chunks.
        Paragraph boundaries first, then sentence boundaries for long paragraphs.
        Filters out example dialogues and conversational samples.
        """
        chunks = []
        for para in text.split("\n"):
            para = para.strip()
            if len(para) < self.MIN_LENGTH:
                continue
            if self._is_example_dialogue(para):
                continue
            if len(para) <= 400:
                chunks.append(para)
            else:
                # Split long paragraph at sentence boundaries
                sentences = para.replace(". ", ".\n").split("\n")
                current = ""
                for s in sentences:
                    if len(current) + len(s) < 400:
                        current += (" " if current else "") + s
                    else:
                        if len(current) >= self.MIN_LENGTH:
                            chunks.append(current.strip())
                        current = s
                if len(current) >= self.MIN_LENGTH:
                    chunks.append(current.strip())

        # Post-filter: catch dialogues that span sentence-split chunks
        chunks = [c for c in chunks if not self._is_example_dialogue(c)]
        return chunks[:self.MAX_CHUNKS]

    def _is_example_dialogue(self, text):
        """
        Detect example conversation / dialogue chunks from dictionary sources.
        These are sample usage, not knowledge — they shouldn't become cells.
        """
        import re
        # Em-dash dialogue turns (e.g., "Anna: Hi! \u2014 Pete: Hello!")
        if text.count("\u2014") >= 2 and ":" in text:
            return True
        if text.count(" \u2014 ") >= 2:
            return True
        # Multiple named speaker labels (e.g., "Anna:", "Pete:")
        speakers = re.findall(r'\b[A-Z][a-z]{1,15}:', text)
        if len(speakers) >= 2:
            return True
        # Stage-direction style quotes with multiple speakers
        if text.count('\u201c') >= 2 or text.count('\u201d') >= 2:
            speaker_count = len(re.findall(r'[A-Z][a-z]+\s*:', text))
            if speaker_count >= 2:
                return True
        return False

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def _ingest(self, substrate, target_lobe, chunks, anchor, source_name):
        """
        Encode each chunk as a LivingCell staged in think lobe.
        cell_id: meth_think_inquisition_{uid}  — ONE copy regardless of target_lobe.
        target_lobe is stored on the cell for later promotion.
        Returns list of new cells created.
        """
        from living_cell import LivingCell

        new_cells = []
        for text in chunks:
            uid     = hashlib.md5(text.encode()).hexdigest()[:12]
            cell_id = f"meth_think_inquisition_{uid}"

            if cell_id in substrate.methodology_cells:
                # Already staged — still useful as a match, return existing
                new_cells.append(substrate.methodology_cells[cell_id])
                continue

            dna  = np.array(substrate.hdc.encode(text), dtype=np.float32)
            norm = np.linalg.norm(dna)
            if norm < 1e-8:
                continue
            dna = dna / norm

            cell = LivingCell(
                cell_id,
                content=text,
                source_table=f"inquisition_{source_name}",
                confidence=self.CONFIDENCE,
                source="inquisition",
            )
            cell.dna                      = dna
            cell.anchor                   = anchor[:200]
            cell.cell_id                  = cell_id
            cell.meta["target_lobe"]      = target_lobe
            cell.meta["promoted"]         = False

            substrate.methodology_cells[cell_id] = cell
            substrate._persist_methodology_cell(cell)
            new_cells.append(cell)

        return new_cells
