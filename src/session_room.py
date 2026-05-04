"""
Session Room
============
Each conversation with IA happens inside a Room.

A Room is the working memory of a single session:
  - Every exchange is encoded and stored as a session cell
  - A running fingerprint (mean query vector) builds up over the session
  - When the session ends, the Room saves a compact snapshot to disk

On the next session, IA loads the archive and checks whether the current
conversation feels like someone it has spoken to before. If the fingerprint
similarity passes a threshold, prior-session cells get loaded into the think
staging pool at low confidence — they surface naturally through ambient recall.

No direct identification. No questions. The recognition happens the way it
does in real life: a familiar turn of phrase, a topic that keeps returning,
a way of asking that feels known.

Session cell prefix: meth_session_{session_id}_{uid}
Session snapshots:   data_store/sessions/{ts}_{session_id}.json
"""

import os
import json
import hashlib
import time
import uuid
import numpy as np


SESSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_store", "sessions"
)

PRIOR_SIM_THRESHOLD  = 0.60   # fingerprint similarity to declare a prior session match
AMBIENT_CONFIDENCE   = 0.65   # confidence for cells loaded from prior sessions
AMBIENT_MIN_EXCHANGES = 5     # minimum exchanges before deductive check fires
ROOM_CELL_CONFIDENCE = 1.3    # room cells are high-confidence — they are current truth


class Room:

    def __init__(self, substrate):
        os.makedirs(SESSIONS_DIR, exist_ok=True)

        self.session_id    = uuid.uuid4().hex[:12]
        self.ts_start      = int(time.time())
        self.substrate     = substrate
        self.cells         = []          # (cid, cell) tuples — current session cells
        self._q_vecs       = []          # query vectors for fingerprint computation
        self._entities     = set()       # extracted entities from queries
        self._activated    = []          # cell IDs activated each turn (for snapshot)
        self._prior        = None        # matched prior session snapshot (if any)
        self._prior_loaded = False       # ensure we only load prior once

    # ── In-session exchange recording ─────────────────────────────────────────

    def add_exchange(self, query, response, q_vec):
        """
        Record an exchange as a session cell.
        Also updates the running fingerprint and entity set.
        """
        from living_cell import LivingCell

        ts     = int(time.time())
        uid    = hashlib.md5(f"{ts}{query}".encode()).hexdigest()[:10]
        cid    = f"meth_session_{self.session_id}_{uid}"
        content = f"{query.strip()}\n{response.strip()[:200]}"

        cell = LivingCell(
            cid,
            content=content,
            source_table="session_room",
            confidence=ROOM_CELL_CONFIDENCE,
            source="session",
        )
        cell.dna     = np.array(q_vec, dtype=np.float32)
        cell.anchor  = query[:200]
        cell.cell_id = cid

        self.substrate.methodology_cells[cid] = cell
        self.cells.append((cid, cell))

        # Update fingerprint
        self._q_vecs.append(np.array(q_vec, dtype=np.float32))

        # Extract entities — capitalised words, not at sentence start
        self._extract_entities(query)

        # Deductive prior check — fires once after enough context accumulates
        if not self._prior_loaded and len(self._q_vecs) >= AMBIENT_MIN_EXCHANGES:
            self._check_prior()

    def query(self, q_vec, threshold=0.60, top_k=4):
        """
        Return session cells relevant to the current query.
        Excludes the very last cell (just added — not meaningful yet as context).
        """
        if len(self.cells) < 2:
            return []

        q = np.array(q_vec, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn < 1e-8:
            return []
        q = q / qn

        scored = []
        for cid, cell in self.cells[:-1]:   # skip most recent
            dna = getattr(cell, "dna", None)
            if dna is None:
                continue
            v  = np.array(dna, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            sim = float(np.dot(q, v / vn))
            if sim >= threshold:
                scored.append((sim, cid, cell))

        scored.sort(reverse=True)
        return scored[:top_k]

    def note_activated(self, cell_ids):
        """Record which cells were activated this turn for the session snapshot."""
        self._activated.extend(cell_ids)

    # ── Session fingerprint ───────────────────────────────────────────────────

    def get_fingerprint(self):
        """Mean of all query vectors so far — the session's semantic centre of gravity."""
        if not self._q_vecs:
            return None
        mean = np.mean(self._q_vecs, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-8 else mean

    # ── Session closure ───────────────────────────────────────────────────────

    def close(self):
        """
        Save session snapshot to disk.
        Called on exit. Lightweight — just the fingerprint, metadata, and cell IDs.
        """
        fp = self.get_fingerprint()
        if fp is None or len(self.cells) == 0:
            return   # nothing worth saving

        snapshot = {
            "session_id":       self.session_id,
            "ts_start":         self.ts_start,
            "ts_end":           int(time.time()),
            "exchanges":        len(self.cells),
            "fingerprint":      fp.tolist(),
            "entities":         sorted(self._entities),
            "activated_cells":  list(set(self._activated))[-60:],   # cap at 60
            "session_cells":    [cid for cid, _ in self.cells],
            "prior_matched":    self._prior.get("session_id") if self._prior else None,
        }

        fname = os.path.join(SESSIONS_DIR, f"{self.ts_start}_{self.session_id}.json")
        try:
            with open(fname, "w") as f:
                json.dump(snapshot, f, indent=2)
        except Exception:
            pass

    # ── Archive ───────────────────────────────────────────────────────────────

    @staticmethod
    def load_archive():
        """
        Load all session snapshots from disk.
        Returns list of snapshot dicts (lightweight — no cell objects).
        """
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        archive = []
        for fname in sorted(os.listdir(SESSIONS_DIR)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(SESSIONS_DIR, fname)) as f:
                    snap = json.load(f)
                archive.append(snap)
            except Exception:
                continue
        return archive

    # ── Deductive prior matching ──────────────────────────────────────────────

    def _check_prior(self):
        """
        Fires once, after AMBIENT_MIN_EXCHANGES.
        Compares current fingerprint against archive.
        If a match is found, loads activated cells from that session
        into the think staging pool at AMBIENT_CONFIDENCE — they
        surface naturally when relevant topics arise.
        """
        self._prior_loaded = True
        fp = self.get_fingerprint()
        if fp is None:
            return

        archive = Room.load_archive()
        best_sim  = 0.0
        best_snap = None

        for snap in archive:
            if snap.get("session_id") == self.session_id:
                continue   # skip current session
            raw_fp = snap.get("fingerprint")
            if not raw_fp:
                continue
            v  = np.array(raw_fp, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            sim = float(np.dot(fp, v / vn))
            if sim > best_sim:
                best_sim  = sim
                best_snap = snap

        if best_sim >= PRIOR_SIM_THRESHOLD and best_snap:
            self._prior = best_snap
            self._load_prior_ambient(best_snap)

    def _load_prior_ambient(self, snap):
        """
        Load activated cells from a prior session into think staging.
        Cells already in the substrate get their confidence gently boosted.
        Cells that were session-only (meth_session_*) are skipped — they
        weren't meant to persist beyond their session.
        """
        loaded = 0
        for cid in snap.get("activated_cells", []):
            if cid.startswith("meth_session_"):
                continue   # session cells don't cross sessions
            cell = self.substrate.methodology_cells.get(cid)
            if cell is None:
                continue
            # Gentle ambient boost — prior relevance is real but unconfirmed
            current_conf = getattr(cell, "confidence", 1.0)
            cell.confidence = min(current_conf + 0.05, 2.0)
            loaded += 1

        self._prior_ambient_count = loaded

    # ── Entity extraction ─────────────────────────────────────────────────────

    def _extract_entities(self, text):
        """
        Lightweight entity extraction — capitalised words not at sentence start,
        and any word over 6 chars that appears more than once across queries.
        """
        words = text.split()
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:\"'()")
            if len(clean) < 3:
                continue
            # Capitalised but not first word of sentence
            if i > 0 and clean[0].isupper() and clean.isalpha():
                self._entities.add(clean)
