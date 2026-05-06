#!/usr/bin/env python3
"""
admin_chat_tee.py — Tee's management console.

Manage Tee's memory: view products, delete products, inspect cells,
and drop into a live sales chat to test changes immediately.

Usage:
    python admin_chat_tee.py
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

# ── Shared helpers (copied from chat_tee.py) ─────────────────────────────────

VAGUE_TOKENS = {
    'how', 'why', 'what', 'really', 'okay', 'ok', 'sure', 'right',
    'and', 'so', 'hm', 'hmm', 'interesting', 'tell me more', 'go on',
    'explain', 'elaborate', 'continue', 'more', 'yes', 'no', 'seriously',
    'makes no sense', 'i see', 'got it', 'and then', 'next', 'after that',
}

def _is_vague(text):
    words = text.lower().split()
    if len(words) <= 3:
        return True
    return text.lower().strip() in VAGUE_TOKENS

def build_query(user_msg, history):
    if _is_vague(user_msg) and history:
        last_tee = history[-1]['tee']
        context = last_tee[:60].strip().rstrip('.,;')
        return f"{context} {user_msg}"
    return user_msg

def get_products(bridge):
    products = set()
    for cell in bridge.substrate.methodology_cells.values():
        t = getattr(cell, 'source_table', '')
        if t.startswith('meth_product_'):
            products.add(t[len('meth_product_'):])
    return sorted(products)

def pretty_product(slug):
    return ' '.join(w.upper() if len(w) <= 4 else w.capitalize()
                    for w in slug.split('_'))

def tee_respond(bridge, user_msg, history, used_cells, product_scope, debug=False):
    query = build_query(user_msg, history)
    hits  = bridge.top_cells(query, k=20)
    if not hits:
        return "Let me think about that.", ""

    def _cell_allowed(cell):
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

    scoped_hits = [(s, cid, c) for s, cid, c in hits if _cell_allowed(c)]
    if not scoped_hits:
        scoped_hits = hits

    used_set    = set(used_cells)
    fresh_hits  = [(s, cid, c) for s, cid, c in scoped_hits if cid not in used_set]
    active_hits = fresh_hits if fresh_hits else scoped_hits
    best_sim, best_cid, best_cell = active_hits[0]

    if debug:
        src     = getattr(best_cell, 'source_table', '?')
        snippet = (getattr(best_cell, 'content', '') or '')[:55].strip()
        q_disp  = query if query != user_msg else user_msg
        print(f"  \033[90m[q={q_disp!r:.55}]")
        print(f"   sim={best_sim:.3f} | {src} | \"{snippet}...\"]\033[0m")

    if best_sim < 0.20:
        return "Let me think about that.", ""

    source = getattr(best_cell, 'source_table', '')
    if 'product' in source:
        response = (getattr(best_cell, 'content', '') or '').strip()
        return (response or "Let me think about that."), best_cid

    response = None
    if bridge._dec is not None and bridge._enc is not None:
        response = bridge._generate(query, temperature=0.75, top_k=40, max_new=80)
        if not response or len(response.strip()) < 8:
            response = None
    if not response:
        response = (getattr(best_cell, 'content', '') or '').strip()
    return (response or "Let me think about that."), best_cid

def tee_print(text):
    print(f"  \033[1mTee:\033[0m {textwrap.fill(text, width=W-6, subsequent_indent='       ')}\n")

def select_product(bridge):
    products = get_products(bridge)
    intro = (
        "Hi, I'm Tee — a sales agent. I'm here to help you find the right "
        "solution for your business. I can walk you through what we offer, "
        "handle your questions, and help you figure out if it's a good fit."
    )
    tee_print(intro)
    if not products:
        tee_print("I don't have any specific products loaded yet, but ask me anything and I'll do my best.")
        return None
    if len(products) == 1:
        slug = products[0]
        tee_print(f"Right now I manage one product: {pretty_product(slug)}. Let me tell you about it.")
        return slug
    names    = [pretty_product(p) for p in products]
    list_str = ', '.join(names[:-1]) + f', and {names[-1]}'
    tee_print(f"I currently manage {len(products)} products: {list_str}. Which one would you like to hear about?")
    print('─' * W)
    for i, (slug, name) in enumerate(zip(products, names), 1):
        print(f"    {i}. {name}")
    print()
    valid = [str(i) for i in range(1, len(products) + 1)]
    while True:
        raw = input("  You: ").strip()
        if raw in valid:
            chosen = products[int(raw) - 1]
            break
        match = [p for p in products if raw.lower() in p.lower()]
        if len(match) == 1:
            chosen = match[0]
            break
        print(f"  Please enter a number (1-{len(products)}) or the product name.")
    print()
    tee_print(f"Great — let's talk about {pretty_product(chosen)}. What would you like to know?")
    return chosen


# ── Admin functions ───────────────────────────────────────────────────────────

def hr():
    print('─' * W)

def banner(text):
    print(f"\n{'='*W}")
    print(f"  {text}")
    print(f"{'='*W}\n")

def ask(prompt, options=None):
    while True:
        raw = input(f"  {prompt} ").strip()
        if raw:
            if options and raw not in options:
                print(f"  Please enter one of: {', '.join(options)}")
                continue
            return raw
        print("  (Please enter a value)")

def ask_yn(prompt):
    ans = ask(f"{prompt} (yes / no)", options=['yes', 'no', 'y', 'n'])
    return ans.lower() in ('yes', 'y')


def admin_list_products(bridge):
    """Show all products and their cell counts."""
    hr()
    print("\n  Products in Tee's memory:\n")
    products = get_products(bridge)

    # Also count legacy flat product cells
    legacy = sum(
        1 for c in bridge.substrate.methodology_cells.values()
        if getattr(c, 'source_table', '') == 'meth_product'
    )

    if not products and not legacy:
        print("  No products loaded yet.\n")
        return

    for slug in products:
        table = f'meth_product_{slug}'
        count = sum(
            1 for c in bridge.substrate.methodology_cells.values()
            if getattr(c, 'source_table', '') == table
        )
        print(f"    {pretty_product(slug):<30}  {count:>4} cells   [{table}]")

    if legacy:
        print(f"    {'(legacy product cells)':<30}  {legacy:>4} cells   [meth_product]")

    total = sum(
        1 for c in bridge.substrate.methodology_cells.values()
        if 'product' in getattr(c, 'source_table', '')
    )
    identity = sum(
        1 for c in bridge.substrate.methodology_cells.values()
        if 'identity' in getattr(c, 'source_table', '')
    )
    reasoning = sum(
        1 for c in bridge.substrate.methodology_cells.values()
        if 'reasoning' in getattr(c, 'source_table', '')
    )
    print(f"\n  Identity cells : {identity}")
    print(f"  Reasoning cells: {reasoning}")
    print(f"  Product cells  : {total}")
    print(f"  Total          : {len(bridge.substrate.methodology_cells)}\n")


def admin_inspect_product(bridge):
    """Show a sample of cells from a chosen product."""
    products = get_products(bridge)
    if not products:
        print("\n  No products to inspect.\n")
        return

    hr()
    print("\n  Which product do you want to inspect?\n")
    for i, slug in enumerate(products, 1):
        print(f"    {i}. {pretty_product(slug)}")
    print()

    valid = [str(i) for i in range(1, len(products) + 1)]
    choice = int(ask(f"Enter 1-{len(products)}:", options=valid)) - 1
    slug  = products[choice]
    table = f'meth_product_{slug}'

    cells = [c for c in bridge.substrate.methodology_cells.values()
             if getattr(c, 'source_table', '') == table]

    print(f"\n  {pretty_product(slug)} — {len(cells)} cells\n")
    print("  How many samples to show?")
    n = ask("Number (default 5):") or '5'
    try:
        n = int(n)
    except ValueError:
        n = 5

    for i, cell in enumerate(cells[:n], 1):
        hr()
        desc    = (cell.meta or {}).get('description', '')
        content = (cell.content or '')[:300]
        print(f"\n  Cell {i}:")
        if desc:
            print(f"  Description : {desc[:70]}")
        print(f"  Content     : {content}")
        if len(cell.content or '') > 300:
            print("  [truncated...]")
    print()


def admin_delete_product(bridge):
    """Delete all cells belonging to a product from disk and live substrate."""
    products = get_products(bridge)
    if not products:
        print("\n  No products to delete.\n")
        return

    hr()
    print("\n  Which product do you want to delete?\n")
    for i, slug in enumerate(products, 1):
        table = f'meth_product_{slug}'
        count = sum(
            1 for c in bridge.substrate.methodology_cells.values()
            if getattr(c, 'source_table', '') == table
        )
        print(f"    {i}. {pretty_product(slug):<28}  ({count} cells)")
    print()

    valid = [str(i) for i in range(1, len(products) + 1)]
    choice = int(ask(f"Enter 1-{len(products)}:", options=valid)) - 1
    slug   = products[choice]
    table  = f'meth_product_{slug}'
    name   = pretty_product(slug)

    # Find matching cell IDs
    to_delete = [
        cid for cid, c in bridge.substrate.methodology_cells.items()
        if getattr(c, 'source_table', '') == table
    ]

    print(f"\n  This will permanently delete {len(to_delete)} cells for {name}.")
    print("  This cannot be undone.\n")

    if not ask_yn(f"Delete {name}?"):
        print("\n  Cancelled.\n")
        return

    meth_dir = os.path.join(os.path.dirname(__file__), 'data_store', 'methodology')
    deleted_disk = 0
    deleted_mem  = 0

    for cid in to_delete:
        # Remove from disk
        path = os.path.join(meth_dir, f"{cid}.json")
        if os.path.exists(path):
            os.remove(path)
            deleted_disk += 1
        # Remove from live substrate
        if cid in bridge.substrate.methodology_cells:
            del bridge.substrate.methodology_cells[cid]
            deleted_mem += 1

    hr()
    print(f"\n  Done. {deleted_disk} files deleted from disk.")
    print(f"  {deleted_mem} cells removed from live memory.")
    print(f"  Total cells remaining: {len(bridge.substrate.methodology_cells)}\n")


def admin_chat(bridge, debug=False):
    """Drop into a live sales chat to test Tee's current state."""
    hr()
    print()
    history:    list  = []
    used_cells: deque = deque(maxlen=4)
    product_scope = select_product(bridge)

    print("  (Type 'quit' to return to admin menu)\n")

    while True:
        try:
            raw = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw.lower() in ('quit', 'exit', 'q', 'bye'):
            break
        print()
        response, cid = tee_respond(bridge, raw, history, used_cells,
                                    product_scope, debug=debug)
        if cid:
            used_cells.append(cid)
        history.append({'user': raw, 'tee': response})
        tee_print(response)

    print(f"\n  {len(history)} turns. Returning to admin menu.\n")


# ── Admin menu ────────────────────────────────────────────────────────────────

MENU = [
    ('List products and cell counts',        admin_list_products),
    ('Inspect cells inside a product',       admin_inspect_product),
    ('Delete a product and all its cells',   admin_delete_product),
    ('Test Tee in a live sales chat',        admin_chat),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    banner("TEE — ADMIN CONSOLE")
    print("  Manage Tee's memory: products, cells, and live testing.\n")
    print("  What you can do here:\n")
    for i, (label, _) in enumerate(MENU, 1):
        print(f"    {i}. {label}")
    print(f"    q. Quit\n")
    print("  Changes to products and cells take effect immediately.")
    print("  You do not need to restart Tee after making changes here.\n")
    hr()

    print("\n  [Loading Tee...]\n", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("  ERROR: Tee's substrate failed to load.\n")
        sys.exit(1)
    print(f"  Tee online — {len(bridge.substrate.methodology_cells)} cells loaded.\n")
    hr()

    valid = [str(i) for i in range(1, len(MENU) + 1)] + ['q']

    while True:
        print("\n  What would you like to do?\n")
        for i, (label, _) in enumerate(MENU, 1):
            print(f"    {i}. {label}")
        print(f"    q. Quit\n")

        choice = ask("Enter choice:", options=valid)
        if choice == 'q':
            print("\n  Goodbye.\n")
            break

        _, fn = MENU[int(choice) - 1]
        if fn == admin_chat:
            fn(bridge, debug=args.debug)
        else:
            fn(bridge)

if __name__ == '__main__':
    main()
