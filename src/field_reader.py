"""
Field Reader — Substrate-Emergent Generation
=============================================
The substrate IS the language model.
The activation field IS the thought.
The walk through the field IS the sentence.

No templates. No training. No frozen decoder.
The response evolves as the cells evolve.

How it works:
  1. Every cell in the substrate is scored against the query (vectorized).
     Cells above MIN_FIELD_SIM form the activation field.
  2. Starting from the most query-aligned activated cell, walk the
     connection graph — at each step, follow the strongest connection
     that is also inside the activation field.
  3. The entry cell is checked for query-mirroring (the prompt-side of a
     connected pair). If it mirrors the query, skip its phrase — use only
     what the connection leads to.
  4. From each walked cell, extract the sentence whose words most overlap
     with the query — the cell's most relevant contribution.
  5. Word-overlap deduplication prevents near-identical phrases stacking.

Walk discipline:
  - Connection-guided steps are preferred and unlimited within MAX_WALK_STEPS.
  - Field jumps (unguided) are capped at 1 and require MIN_JUMP_ACTIVATION.
    This prevents the walk straying into unrelated high-activation cells.

As cells grow (energy, confidence, connections), the field changes.
As cells are added, new attractors appear in the field.
The walk finds new paths. The voice evolves.
"""

import re
import numpy as np

SKIP_CONTENT_PREFIXES = (
    "{", "[{", "meth_", "def ", "import ", "class ", "SELECT ",
    "Write me", "write me", "Conversational Transition",
    "poem,", "Poem,",
    "do you dream", "Do you dream",
)
SKIP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "i", "it", "its", "this", "that", "these", "those", "there", "here",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "and",
    "or", "but", "so", "yet", "not", "no", "yes", "can", "could", "would",
    "should", "will", "do", "does", "did", "have", "has", "had",
}


def _load_generator():
    try:
        from neural.ia_generator import IAGenerator
        return IAGenerator(device='cpu')
    except Exception:
        return None


class FieldReader:

    MIN_FIELD_SIM       = 0.28   # minimum cosine sim to enter the field
    PHRASE_MIN_LEN      = 15     # minimum characters for a usable phrase
    MAX_WALK_STEPS      = 4      # max cells to visit total
    MAX_FIELD_JUMPS     = 1      # max unguided field jumps per walk
    MIN_JUMP_ACTIVATION = 0.50   # minimum activation for an unguided jump
    CONTENT_MAX         = 450    # max chars to read from a cell's content
    MIRROR_OVERLAP      = 0.45   # word-overlap threshold for query-mirror detection
    MIRROR_CONN_MIN     = 0.55   # connection strength that confirms mirror role
    DEDUP_OVERLAP       = 0.45   # word-overlap threshold for phrase deduplication

    def __init__(self):
        self.last_field_size = 0
        self.last_walk_len   = 0
        self._generator      = _load_generator()

    def read(self, substrate, query_vec, query_text, activated_cells):
        """
        Score the full substrate, walk the activation field, extract one
        phrase per walked cell. Returns the composed response string.

        Side-effects: sets self.last_field_size and self.last_walk_len
        for display purposes.
        """
        field = self._build_field(substrate, query_vec)
        self.last_field_size = len(field)

        if not field and not activated_cells:
            self.last_walk_len = 0
            return "I don't have enough to go on yet."

        query_words = self._words(query_text)
        conj_vec    = self._conjugate_vec(substrate, query_vec)

        walk = self._walk(substrate, field, activated_cells, query_words, conj_vec)
        self.last_walk_len = len(walk)

        phrases    = []
        seen_wsets = []   # word sets of already-included phrases (for overlap dedup)

        for cell in walk:
            phrase = self._extract_phrase(cell, query_words)
            if not phrase:
                continue

            # Questions are prompt-side content, not responses
            if phrase.rstrip().endswith("?"):
                continue

            p_words = self._words(phrase)

            # Skip phrases that echo the query — A-side bleed-through
            if query_words:
                echo = len(p_words & query_words) / max(len(query_words), 1)
                if echo >= 0.55:
                    continue

            # Word-overlap dedup — don't stack near-identical ideas
            duplicate = any(
                len(p_words & sw) / max(len(p_words), 1) >= self.DEDUP_OVERLAP
                for sw in seen_wsets
            )
            if duplicate:
                continue

            seen_wsets.append(p_words)
            phrases.append(phrase)

        # Neural generation — try decoder first if trained
        if self._generator and self._generator.ready:
            generated = self._generator.compose(walk, conj_vec)
            if generated:
                return generated

        # Short queries (greetings, single words) — one phrase is enough
        if len(query_words) <= 1 and phrases:
            return phrases[0]

        if not phrases:
            # Pass 1: follow mirror connections to find the response-side cell
            for _, _, cell in activated_cells:
                if not self._is_query_mirror(cell, query_words):
                    continue
                connections = getattr(cell, "connections", None) or {}
                for conn_cid, strength in sorted(connections.items(), key=lambda x: -x[1]):
                    if strength < self.MIRROR_CONN_MIN:
                        break
                    conn_cell = substrate.methodology_cells.get(conn_cid)
                    if not conn_cell:
                        continue
                    if self._is_query_mirror(conn_cell, query_words):
                        continue
                    phrase = self._extract_phrase(conn_cell, query_words)
                    if phrase and not phrase.rstrip().endswith("?"):
                        if not query_words or len(self._words(phrase) & query_words) / max(len(query_words), 1) < 0.55:
                            return phrase

            # Pass 2: first non-mirror cell with usable declarative content
            for _, _, cell in activated_cells:
                if self._is_query_mirror(cell, query_words):
                    continue
                phrase = self._extract_phrase(cell, query_words)
                if phrase and not phrase.rstrip().endswith("?"):
                    if not query_words or len(self._words(phrase) & query_words) / max(len(query_words), 1) < 0.55:
                        return phrase

            # Pass 3: scan full field sorted by activation — catches single-word
            # queries where lobes return only mirror cells
            for fcid in sorted(field, key=field.get, reverse=True):
                fcell = substrate.methodology_cells.get(fcid)
                if not fcell:
                    continue
                if self._is_query_mirror(fcell, query_words):
                    continue
                phrase = self._extract_phrase(fcell, query_words)
                if phrase and not phrase.rstrip().endswith("?"):
                    if not query_words or len(self._words(phrase) & query_words) / max(len(query_words), 1) < 0.55:
                        return phrase

            return "I'm still working on that one."

        return "\n\n".join(phrases)

    # ── Field scoring ──────────────────────────────────────────────────────────

    def _build_field(self, substrate, query_vec):
        """
        Score every cell in the substrate against the query (vectorized).
        Returns {cid: activation} for all cells above MIN_FIELD_SIM.
        Activation = sim × confidence × energy (capped at 1.5).
        """
        cells_list = []
        vecs       = []

        for cid, cell in substrate.methodology_cells.items():
            if cid.startswith("meth_mind_console_"):
                continue
            dna = getattr(cell, "dna", None)
            if dna is None:
                continue
            v = np.array(dna, dtype=np.float32)
            pull = getattr(cell, "_pull", None)
            if pull is not None:
                v = v + pull
            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            cells_list.append((cid, cell))
            vecs.append(v / vn)

        if not vecs:
            return {}

        matrix = np.stack(vecs, axis=0)    # (N, 384)
        sims   = matrix @ query_vec        # (N,) cosine similarity

        field = {}
        for i, (cid, cell) in enumerate(cells_list):
            sim = float(sims[i])
            if sim < self.MIN_FIELD_SIM:
                continue
            activation = (
                sim
                * getattr(cell, "confidence", 1.0)
                * min(getattr(cell, "energy", 1.0), 1.5)
            )
            field[cid] = activation

        return field

    def _conjugate_vec(self, substrate, query_vec):
        """
        Compute the conjugate direction: 2G - Q where G is the substrate centroid.

        The centroid is the mean of all normalized cell DNA vectors — the resting
        state of the substrate. Reflecting the query through the centroid points
        toward the response-side attractor: the vector space where answers live.
        """
        vecs = []
        for cid, cell in substrate.methodology_cells.items():
            if cid.startswith("meth_mind_console_"):
                continue
            dna = getattr(cell, "dna", None)
            if dna is None:
                continue
            v = np.array(dna, dtype=np.float32)
            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            vecs.append(v / vn)
        if not vecs:
            return query_vec
        centroid = np.mean(vecs, axis=0)
        cn = np.linalg.norm(centroid)
        if cn < 1e-8:
            return query_vec
        centroid = centroid / cn
        raw = 2.0 * centroid - query_vec
        rn = np.linalg.norm(raw)
        return raw / rn if rn > 1e-8 else centroid

    # ── Walk ──────────────────────────────────────────────────────────────────

    def _walk(self, substrate, field, activated_cells, query_words, conj_vec=None):
        """
        Walk the activation field starting from the conjugate-aligned seed cell.

        Seed selection uses the conjugate vector (2G - Q) to skip mirror cells
        and land directly on the response-side of the field. Falls back to
        highest-activation cell if no conjugate seed is found.

        Mirror cells (prompt-side of a pair) are followed for their connections
        but never added to walk[] — they ARE the query, not the answer.

        At each step:
          1. Follow the strongest connection that is inside the field.
             If current is a mirror, ONLY follow connections — no field jump.
          2. If no connection leads forward and current is not a mirror,
             take one field-guided jump to a non-mirror cell above
             MIN_JUMP_ACTIVATION. After that, stop.
        """
        visited     = set()
        walk        = []
        field_jumps = 0

        if not activated_cells:
            if not field:
                return []
            top_cid = max(field, key=field.get)
            cell    = substrate.methodology_cells.get(top_cid)
            if cell is None:
                return []
            activated_cells = [(field[top_cid], top_cid, cell)]

        # Conjugate seed: score field cells against conj_vec, pick highest
        # non-mirror cell — this is the response-side attractor
        current = None
        if conj_vec is not None and field:
            best_conj_score = -1.0
            for fcid, act in field.items():
                fcell = substrate.methodology_cells.get(fcid)
                if fcell is None:
                    continue
                if self._is_query_mirror(fcell, query_words):
                    continue
                dna = getattr(fcell, "dna", None)
                if dna is None:
                    continue
                v = np.array(dna, dtype=np.float32)
                pull = getattr(fcell, "_pull", None)
                if pull is not None:
                    v = v + pull
                vn = np.linalg.norm(v)
                if vn < 1e-8:
                    continue
                score = float((v / vn) @ conj_vec)
                if score > best_conj_score:
                    best_conj_score = score
                    current = fcell

        # Fallback: highest-activation cell from activated_cells
        if current is None:
            _, _, current = max(activated_cells, key=lambda x: x[0])

        for _ in range(self.MAX_WALK_STEPS):
            cid = getattr(current, "cell_id", str(id(current)))
            if cid in visited:
                break
            visited.add(cid)

            is_mirror = self._is_query_mirror(current, query_words)

            # Only collect content from non-mirror cells
            if not is_mirror:
                content = str(getattr(current, "content", "")).strip()
                if self._valid(content):
                    walk.append(current)

            # Step 1: connection-guided advance
            connections = getattr(current, "connections", None) or {}
            best_next   = None
            best_score  = 0.0

            for conn_cid, strength in connections.items():
                if conn_cid in visited:
                    continue
                activation = field.get(conn_cid, 0.0)
                if activation < 0.05:
                    continue
                score = strength * (1.0 + activation)
                if score > best_score:
                    best_score = score
                    best_next  = conn_cid

            if best_next and best_next in substrate.methodology_cells:
                current = substrate.methodology_cells[best_next]
                continue

            # Mirror cells with no forward connection are a dead end — stop
            if is_mirror:
                break

            # Step 2: one field-guided jump, high bar, non-mirror cells only
            if field_jumps >= self.MAX_FIELD_JUMPS:
                break

            best_cid = None
            best_act = 0.0
            for fcid, act in field.items():
                if fcid in visited:
                    continue
                if act < self.MIN_JUMP_ACTIVATION:
                    continue
                candidate = substrate.methodology_cells.get(fcid)
                if candidate and self._is_query_mirror(candidate, query_words):
                    continue
                if act > best_act:
                    best_act = act
                    best_cid = fcid

            if best_cid:
                current = substrate.methodology_cells[best_cid]
                field_jumps += 1
                continue
            break

        return walk

    # ── Phrase extraction ──────────────────────────────────────────────────────

    def _extract_phrase(self, cell, query_words):
        """
        Extract the single sentence from a cell's content that best
        expresses the cell's contribution to the current query.

        Scored by word overlap with the query (stop-words excluded).
        No encoding calls — the field scoring already paid for similarity.
        """
        content = str(getattr(cell, "content", "")).strip()
        if not self._valid(content):
            return None

        content   = content[:self.CONTENT_MAX]
        sentences = self._split_sentences(content)

        if not sentences:
            return content if len(content) >= self.PHRASE_MIN_LEN else None
        if len(sentences) == 1:
            return sentences[0]

        # Prefer declarative sentences — questions echo the query, not answer it
        # Also strip any sentence that starts with a known skip prefix
        declarative = [
            s for s in sentences
            if not s.rstrip().endswith("?")
            and not any(s.startswith(p) for p in SKIP_CONTENT_PREFIXES)
        ]
        candidates  = declarative if declarative else sentences

        if len(candidates) == 1:
            return candidates[0]

        # Score by word overlap with query among declarative candidates
        best_score    = -1.0
        best_sentence = candidates[0]
        for s in candidates:
            s_words = self._words(s)
            overlap = len(s_words & query_words)
            score   = overlap / max(len(s_words), 1)
            if score > best_score:
                best_score    = score
                best_sentence = s

        # If the winner still echoes the query heavily, fall back to longest sentence
        # (the substantive response is usually the fuller one)
        winner_words = self._words(best_sentence)
        echo_ratio   = len(winner_words & query_words) / max(len(winner_words), 1)
        if echo_ratio > 0.55 and len(candidates) > 1:
            best_sentence = max(candidates, key=len)

        return best_sentence

    # ── Query-mirror detection ─────────────────────────────────────────────────

    def _is_query_mirror(self, cell, query_words):
        """
        Returns True if this cell looks like the prompt-side of a pair.
        Two cases:

        1. Standard mirror — content word-overlap with query ≥ MIRROR_OVERLAP
           AND has a strong forward connection (it's a wired exchange A-side).

        2. Prefix mirror — content OPENS with most of the query words
           (concatenated A+B cell from console log wiring). Same connection
           requirement.

        Connection strength ≥ MIRROR_CONN_MIN confirms the cell leads
        somewhere — otherwise it might just be a topically similar response.
        """
        if not query_words:
            return False

        content = str(getattr(cell, "content", "")).strip()
        if not content:
            return False

        connections = getattr(cell, "connections", None) or {}
        has_strong_conn = any(s >= self.MIRROR_CONN_MIN for s in connections.values())
        if not has_strong_conn:
            return False

        content_words = self._words(content)
        if not content_words:
            return False

        # Standard: content as a whole overlaps the query heavily
        overlap = len(content_words & query_words) / max(len(content_words), 1)
        if overlap >= self.MIRROR_OVERLAP:
            return True

        # Prefix: opening words of content match most of the query
        # (catches "Tell me something interesting I'm glad to report..." cells)
        opening = self._words(" ".join(content.split()[:len(query_words) + 3]))
        prefix_overlap = len(opening & query_words) / max(len(query_words), 1)
        return prefix_overlap >= 0.60

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split_sentences(self, text):
        """Split text into sentences, filter short ones."""
        parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
        return [p.strip() for p in parts if len(p.strip()) >= self.PHRASE_MIN_LEN]

    def _words(self, text):
        """Lowercase word set, stop-words removed."""
        return {w for w in re.findall(r'\w+', text.lower()) if w not in SKIP_WORDS and len(w) > 2}

    def _valid(self, content):
        if len(content.strip()) < self.PHRASE_MIN_LEN:
            return False
        for prefix in SKIP_CONTENT_PREFIXES:
            if content.startswith(prefix):
                return False
        # Reject verse fragments: 3+ consecutive short lines without terminal punctuation
        lines = [l for l in content.splitlines() if l.strip()]
        if len(lines) >= 3:
            short_unpunctuated = sum(
                1 for l in lines
                if len(l.strip()) < 60 and not l.rstrip().endswith(('.', '!', '?', ':'))
            )
            if short_unpunctuated >= 3:
                return False
        return True
