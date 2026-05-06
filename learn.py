"""
learn.py — Tee's live learning layer.

Three responsibilities:
  1. SessionLogger  — writes every conversation turn to data_store/sessions/
  2. MissLogger     — writes every sub-threshold query to data_store/misses/
  3. distill()      — reads session logs and extracts high-confidence cell candidates

Called from chat_tee.py and talk_tee.py every turn. Zero overhead when nothing fires.

Session log format  (data_store/sessions/session_<id>.jsonl):
  one JSON object per line, one line per turn:
  {"ts": 1234, "user": "...", "tee": "...", "sim": 0.87, "cell_id": "...", "product": "voxi"}

Miss log format  (data_store/misses/miss_log.jsonl):
  {"ts": 1234, "query": "...", "top_sim": 0.11, "product": "voxi"}

Candidate format  (data_store/candidates/candidates.jsonl):
  {"ts": 1234, "description": "...", "content": "...", "source_table": "...",
   "avg_sim": 0.91, "session_id": "...", "status": "pending"}
"""

import os
import json
import time
import uuid

_ROOT      = os.path.dirname(os.path.abspath(__file__))
_SESS_DIR  = os.path.join(_ROOT, 'data_store', 'sessions')
_MISS_DIR  = os.path.join(_ROOT, 'data_store', 'misses')
_CAND_DIR  = os.path.join(_ROOT, 'data_store', 'candidates')

for _d in (_SESS_DIR, _MISS_DIR, _CAND_DIR):
    os.makedirs(_d, exist_ok=True)

MISS_LOG  = os.path.join(_MISS_DIR,  'miss_log.jsonl')
CAND_FILE = os.path.join(_CAND_DIR,  'candidates.jsonl')

# Thresholds
SIM_STRONG   = 0.50   # high-confidence fire — candidate for distillation
SIM_WEAK     = 0.20   # minimum to fire at all
SIM_MISS     = 0.20   # below this → miss log


# ── Session Logger ────────────────────────────────────────────────────────────

class SessionLogger:
    """
    One instance per conversation session.
    Call .log() after every Tee response.
    Call .close() at session end (writes summary line).
    """

    def __init__(self, product: str | None = None):
        self.session_id = uuid.uuid4().hex[:12]
        self.product    = product
        self.turns      = 0
        self._path      = os.path.join(_SESS_DIR, f"session_{self.session_id}.jsonl")
        self._fh        = open(self._path, 'w', encoding='utf-8')

    def log(self, user: str, tee: str, sim: float, cell_id: str):
        record = {
            'ts':       time.time(),
            'user':     user,
            'tee':      tee,
            'sim':      round(sim, 4),
            'cell_id':  cell_id,
            'product':  self.product,
        }
        self._fh.write(json.dumps(record) + '\n')
        self._fh.flush()
        self.turns += 1

    def close(self):
        summary = {
            'ts':         time.time(),
            'type':       'summary',
            'session_id': self.session_id,
            'turns':      self.turns,
            'product':    self.product,
        }
        self._fh.write(json.dumps(summary) + '\n')
        self._fh.close()


# ── Miss Logger ───────────────────────────────────────────────────────────────

def log_miss(query: str, top_sim: float, product: str | None):
    """
    Write a sub-threshold query to the miss log.
    Called when sim < SIM_MISS — Tee had no good answer.
    """
    record = {
        'ts':      time.time(),
        'query':   query,
        'top_sim': round(top_sim, 4),
        'product': product,
    }
    with open(MISS_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


# ── Distiller ─────────────────────────────────────────────────────────────────

def distill(min_sim: float = SIM_STRONG, dry_run: bool = False) -> list:
    """
    Read all session logs. Find turns where:
      - sim >= min_sim (Tee fired with high confidence)
      - the conversation continued naturally (next user turn is not empty)

    Extract as candidate cells: the user query becomes the description,
    the Tee response becomes the content.

    Returns list of candidate dicts. Writes to candidates.jsonl unless dry_run.
    """
    session_files = [
        os.path.join(_SESS_DIR, f)
        for f in os.listdir(_SESS_DIR)
        if f.startswith('session_') and f.endswith('.jsonl')
    ]

    # Load existing candidates to avoid duplicates
    existing = set()
    if os.path.exists(CAND_FILE):
        with open(CAND_FILE, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing.add(r.get('description', ''))
                except Exception:
                    pass

    candidates = []

    for path in session_files:
        try:
            turns = []
            with open(path, encoding='utf-8') as f:
                for line in f:
                    try:
                        turns.append(json.loads(line))
                    except Exception:
                        pass

            # Only look at non-summary turns
            data_turns = [t for t in turns if t.get('type') != 'summary']

            for i, turn in enumerate(data_turns):
                sim     = turn.get('sim', 0.0)
                user    = turn.get('user', '').strip()
                tee     = turn.get('tee', '').strip()
                product = turn.get('product')

                if sim < min_sim:
                    continue
                if not user or not tee:
                    continue
                if user in existing:
                    continue
                # Skip identity responses — already well covered
                if len(user.split()) <= 2:
                    continue

                # Determine source table from product scope
                if product:
                    source_table = f'product_{product}'
                else:
                    source_table = 'conversation'

                candidate = {
                    'ts':           time.time(),
                    'description':  user,
                    'content':      tee,
                    'source_table': source_table,
                    'avg_sim':      round(sim, 4),
                    'session_id':   os.path.basename(path).replace('session_', '').replace('.jsonl', ''),
                    'status':       'pending',
                }
                candidates.append(candidate)
                existing.add(user)

        except Exception as e:
            print(f"[learn] Skipped {path}: {e}")

    if not dry_run and candidates:
        with open(CAND_FILE, 'a', encoding='utf-8') as f:
            for c in candidates:
                f.write(json.dumps(c) + '\n')

    return candidates


# ── Miss report ───────────────────────────────────────────────────────────────

def miss_report(top_n: int = 20) -> list:
    """
    Read the miss log and return the top-N most frequent unanswered queries.
    Used by review_queue.py to surface what cells to write next.
    """
    if not os.path.exists(MISS_LOG):
        return []

    from collections import Counter
    counts  = Counter()
    records = {}

    with open(MISS_LOG, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                q = r.get('query', '').strip().lower()
                if q:
                    counts[q] += 1
                    records[q] = r
            except Exception:
                pass

    top = counts.most_common(top_n)
    return [{'query': q, 'count': n, 'product': records[q].get('product'),
             'top_sim': records[q].get('top_sim', 0)} for q, n in top]
