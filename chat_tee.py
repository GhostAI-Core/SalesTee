#!/usr/bin/env python3
"""
chat_tee.py — Interactive sales conversation with Tee.

Architecture:
  - Retrieval only: no decoder generation (raw cell content always)
  - Identity questions bypass context enrichment
  - Internal/monologue cells never shown to prospect
  - Product scoping: auto-detected from first message, or Tee lists options
  - Scales cleanly to N products as they are added via ingest_doc.py

Usage:
    python chat_tee.py
    python chat_tee.py --debug
"""

import os
import sys
import textwrap
import argparse
from collections import deque

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

W = 72

# ── Query classification ──────────────────────────────────────────────────────

# These trigger direct identity lookup — no context enrichment applied
IDENTITY_TRIGGERS = {
    'who are you', 'who are you?', 'what are you', 'what is your name',
    'are you human', 'are you a bot', 'are you a robot', 'are you real',
    'are you ai', 'are you an ai', 'who created you', 'what do you do',
    'what is your job', 'what is your philosophy', 'how do you sell',
    'what do you believe', 'what makes you different', 'do you use scripts',
    'how do you qualify', 'can you guarantee', 'what are you good at',
    'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
    'howdy', 'sup', 'what\'s up', 'whats up',
}

# Three-tier response thresholds
SIM_STRONG = 0.40   # Tier 1: fire directly
SIM_WEAK   = 0.20   # Tier 2: fire + clarifying question
# Below SIM_WEAK = Tier 3: honest deflection + miss logged by caller

# Clarifying questions Tee uses when confidence is medium (Tier 2)
_CLARIFY = [
    "Just want to make sure I'm answering the right thing — can you tell me a bit more?",
    "I want to give you the most useful answer — can you be more specific about what you're asking?",
    "Good question. Are you asking from a cost angle, a process angle, or something else?",
    "I've got a few angles I could take on that. What's the main thing you're trying to figure out?",
    "Let me make sure I understand — is this about setup, pricing, or how it works day-to-day?",
]
_clarify_idx = 0

def _next_clarifier() -> str:
    global _clarify_idx
    q = _CLARIFY[_clarify_idx % len(_CLARIFY)]
    _clarify_idx += 1
    return q

# Short/generic inputs that need context from last response to retrieve well
VAGUE_TOKENS = {
    'how', 'why', 'really', 'okay', 'ok', 'sure', 'right', 'and', 'so',
    'hm', 'hmm', 'interesting', 'tell me more', 'go on', 'explain',
    'elaborate', 'continue', 'more', 'seriously', 'makes no sense',
    'i see', 'got it', 'and then', 'next', 'after that',
}


def _classify(text: str) -> str:
    """
    Returns:
      'identity' — direct identity question, no enrichment
      'vague'    — too short/generic, needs context enrichment
      'normal'   — standard retrieval
    """
    clean = text.lower().strip().rstrip('?!')
    if clean in IDENTITY_TRIGGERS:
        return 'identity'
    words = clean.split()
    if len(words) <= 2 or clean in VAGUE_TOKENS:
        return 'vague'
    return 'normal'


def build_query(user_msg: str, history: list, window: int = 3,
                product_scope: str | None = None) -> str:
    """
    Identity questions: query as-is — never pollute with prior context.
    Vague inputs: anchor to last N Tee responses + product name if scoped.
    Normal inputs: prepend product name when scoped to pull DNA toward product space.
    """
    kind = _classify(user_msg)
    if kind == 'identity':
        return user_msg

    product_prefix = product_scope.replace('_', ' ') if product_scope else ''

    if kind == 'vague' and history:
        recent = history[-window:]
        ctx = ' '.join(h['tee'][:80].strip().rstrip('.,;') for h in recent)
        base = f"{ctx} {user_msg}"
    else:
        base = user_msg

    return f"{product_prefix} {base}".strip() if product_prefix else base


# ── Cell filtering ────────────────────────────────────────────────────────────

def _is_prospect_facing(cell, block_patience: bool = False) -> bool:
    """
    Internal/monologue cells are always excluded from prospect responses.
    block_patience=True additionally excludes patience verbal cells — used
    when a product scope is active and we want product content, not filler.
    """
    meta = getattr(cell, 'meta', {}) or {}
    if meta.get('internal', False):
        return False
    desc = meta.get('description', '')
    if desc.startswith('monologue:') or desc.startswith('patience: internal'):
        return False
    if block_patience and desc.startswith('patience:'):
        return False
    return True


def _cell_in_scope(cell, product_scope: str | None) -> bool:
    """
    Always allow identity + reasoning cells.
    Product cells: only allow if they match the active scope.
    If no scope is set, all product cells are allowed (pre-selection state).
    Legacy flat meth_product cells are always allowed as fallback.
    """
    t = getattr(cell, 'source_table', '')
    if not t.startswith('meth_product'):
        return True
    if product_scope is None:
        return True
    if t == f'meth_product_{product_scope}':
        return True
    if t == 'meth_product':
        return True
    return False


# ── Product helpers ───────────────────────────────────────────────────────────

def get_products(bridge) -> list:
    products = set()
    for cell in bridge.substrate.methodology_cells.values():
        t = getattr(cell, 'source_table', '')
        if t.startswith('meth_product_'):
            products.add(t[len('meth_product_'):])
    return sorted(products)


def pretty_product(slug: str) -> str:
    return ' '.join(w.upper() if len(w) <= 4 else w.capitalize()
                    for w in slug.split('_'))


def detect_product_intent(query: str, products: list) -> str | None:
    """Check if the user's first message mentions a known product by name."""
    q = query.lower()
    for slug in products:
        name = pretty_product(slug).lower()
        if slug.replace('_', ' ') in q or name in q:
            return slug
    return None


# ── Core response engine ──────────────────────────────────────────────────────

def tee_respond(bridge, user_msg: str, history: list,
                used_cells: deque, product_scope: str | None,
                debug: bool = False) -> tuple:
    """
    Returns (response_text, fired_cell_id).

    Pipeline:
      1. Classify query — identity / vague / normal
      2. Build retrieval query (enriched or direct)
      3. Retrieve top-K cells
      4. Filter: scope + prospect-facing only
      5. Exclude recently used cells
      6. Return raw content of best hit — no decoder generation
    """
    query = build_query(user_msg, history, product_scope=product_scope)
    hits  = bridge.top_cells(query, k=25)

    if not hits:
        return "Let me think about that.", "", 0.0

    # Block patience verbals when a product scope is active — they fire too
    # easily on vague product questions and sound completely wrong
    block_patience = product_scope is not None
    filtered = [
        (s, cid, c) for s, cid, c in hits
        if _cell_in_scope(c, product_scope) and _is_prospect_facing(c, block_patience)
    ]
    if not filtered:
        # Relax patience block before opening up entirely
        filtered = [
            (s, cid, c) for s, cid, c in hits
            if _cell_in_scope(c, product_scope) and _is_prospect_facing(c)
        ]
    if not filtered:
        filtered = hits  # nothing survived filters — open it up

    # When product-scoped, prefer product cells over reasoning cells if any
    # product cell scored above SIM_WEAK — stops generic queries landing on
    # sales methodology content when there is a relevant product cell available
    if product_scope:
        product_table = f'meth_product_{product_scope}'
        product_hits = [(s, cid, c) for s, cid, c in filtered
                        if getattr(c, 'source_table', '') == product_table
                        and s >= SIM_WEAK]
        if product_hits:
            filtered = product_hits + [
                (s, cid, c) for s, cid, c in filtered
                if getattr(c, 'source_table', '') != product_table
            ]

    # Prefer fresh cells (not fired in last 4 turns)
    used_set   = set(used_cells)
    fresh      = [(s, cid, c) for s, cid, c in filtered if cid not in used_set]
    active     = fresh if fresh else filtered

    best_sim, best_cid, best_cell = active[0]

    if debug:
        src     = getattr(best_cell, 'source_table', '?')
        desc    = (getattr(best_cell, 'meta', {}) or {}).get('description', '')
        q_disp  = query if query != user_msg else user_msg
        print(f"  \033[90m[q={q_disp!r:.55}]")
        print(f"   sim={best_sim:.3f} | {src} | desc={desc!r:.45}]\033[0m")

    # Tier 3: below threshold — honest deflection, miss logged by caller
    if best_sim < SIM_WEAK:
        return (
            "That's outside what I know right now. I'll flag it for the team — "
            "they'll make sure I can answer it properly.",
            "",
            best_sim,
        )

    # Always return raw cell content — no decoder generation
    response = (getattr(best_cell, 'content', '') or '').strip()

    # Tier 2: medium confidence — append a clarifying question to keep it moving
    if best_sim < SIM_STRONG:
        response = f"{response} {_next_clarifier()}"

    return (response or "That's outside what I know right now."), best_cid, best_sim


# ── Display ───────────────────────────────────────────────────────────────────

def tee_print(text: str):
    print(f"  \033[1mTee:\033[0m {textwrap.fill(text, width=W-6, subsequent_indent='       ')}\n")


# ── Opening flow ──────────────────────────────────────────────────────────────

def opening_sequence(bridge, debug: bool) -> str | None:
    """
    Tee introduces herself, then:
      - No products loaded: continues unscoped
      - One product: names it and scopes automatically
      - Multiple products: lists them and lets user choose by name or number
    Returns the chosen product slug (or None).
    """
    products = get_products(bridge)

    tee_print(
        "Hi, I'm Tee — a sales agent. I'm here to figure out what you "
        "need and whether we can actually help. I'll be straight with you "
        "either way. What's on your mind?"
    )

    if not products:
        return None

    if len(products) == 1:
        slug = products[0]
        tee_print(
            f"I'm currently here to talk about {pretty_product(slug)}. "
            f"What would you like to know?"
        )
        return slug

    # Multiple products — let user say what they're interested in
    names = [pretty_product(p) for p in products]
    list_str = ', '.join(names[:-1]) + f' and {names[-1]}'
    tee_print(
        f"I cover {len(products)} products right now: {list_str}. "
        f"Which one are you here about?"
    )

    print('─' * W)
    for i, (slug, name) in enumerate(zip(products, names), 1):
        print(f"    {i}. {name}")
    print()

    while True:
        raw = input("  You: ").strip()
        if not raw:
            continue
        # Number selection
        if raw.isdigit() and 1 <= int(raw) <= len(products):
            chosen = products[int(raw) - 1]
            break
        # Name match
        match = [p for p in products if raw.lower() in p.replace('_', ' ').lower()
                 or p.replace('_', ' ').lower() in raw.lower()]
        if len(match) >= 1:
            chosen = match[0]
            break
        # Not recognised — treat as a question, auto-detect or leave unscoped
        intent = detect_product_intent(raw, products)
        if intent:
            chosen = intent
            break
        tee_print("Which product are you asking about? " +
                  ', '.join(f"{i+1}. {pretty_product(p)}" for i, p in enumerate(products)))
        continue

    print()
    tee_print(f"Let's talk about {pretty_product(chosen)}. What would you like to know?")
    return chosen


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Show retrieval details per turn')
    args = parser.parse_args()

    print(f"\n{'='*W}")
    print(f"  TEE — SALES CHAT")
    print(f"  You are the prospect. Type 'quit' to end.")
    print(f"{'='*W}\n")

    print("[Loading Tee...]", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("ERROR: Tee's substrate failed to load.")
        sys.exit(1)

    cells      = bridge.substrate.methodology_cells
    n_identity = sum(1 for c in cells.values() if 'identity' in getattr(c,'source_table',''))
    n_reasoning= sum(1 for c in cells.values() if 'reasoning' in getattr(c,'source_table',''))
    n_product  = sum(1 for c in cells.values() if 'product' in getattr(c,'source_table',''))
    print(f"[Tee online — {len(cells)} cells: {n_identity} identity / {n_reasoning} reasoning / {n_product} product]\n")
    print('─' * W)
    print()

    history:    list  = []
    used_cells: deque = deque(maxlen=6)

    from learn import SessionLogger, log_miss, SIM_MISS
    session_log = SessionLogger(product=None)

    product_scope = opening_sequence(bridge, debug=args.debug)
    session_log.product = product_scope

    while True:
        try:
            raw = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() in ('quit', 'exit', 'q', 'bye'):
            break

        # On first real message, try to auto-detect product intent if unscoped
        if product_scope is None:
            products = get_products(bridge)
            if products:
                intent = detect_product_intent(raw, products)
                if intent:
                    product_scope = intent
                    session_log.product = product_scope
                    if args.debug:
                        print(f"  [auto-scoped to: {product_scope}]")

        print()
        response, cid, sim = tee_respond(bridge, raw, history, used_cells,
                                         product_scope, debug=args.debug)
        if cid:
            used_cells.append(cid)

        if sim < SIM_MISS:
            log_miss(raw, sim, product_scope)
        else:
            session_log.log(user=raw, tee=response, sim=sim, cell_id=cid)

        history.append({'user': raw, 'tee': response})
        tee_print(response)

    session_log.close()
    print(f"\n{'─'*W}")
    print(f"  {len(history)} turns. Session ended.")
    print(f"{'='*W}\n")


if __name__ == '__main__':
    main()
