#!/usr/bin/env python3
"""
review_queue.py — Human review of Tee's learning candidates.

Three queues:
  1. Candidates  — high-confidence turns distilled from live sessions
  2. Misses      — questions Tee couldn't answer (from miss log)
  3. Weak hits   — turns where sim was 0.20-0.49 (optional, from sessions)

For each item you can:
  a  — approve as-is → saved directly to Tee's substrate
  e  — edit description or content before saving
  s  — skip (leave as pending for later)
  d  — discard permanently

Usage:
    python review_queue.py              # review candidates + misses
    python review_queue.py --candidates # candidates only
    python review_queue.py --misses     # miss log only
    python review_queue.py --distill    # run distiller first, then review
"""

import os
import sys
import json
import time
import argparse
import textwrap

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

W = 76

BOLD  = '\033[1m'
DIM   = '\033[90m'
GREEN = '\033[92m'
YELLOW= '\033[93m'
RED   = '\033[91m'
CYAN  = '\033[96m'
RESET = '\033[0m'


def hr():
    print('─' * W)

def banner(text):
    print(f"\n{'='*W}")
    print(f"  {text}")
    print(f"{'='*W}\n")

def wrap(text, indent=4):
    prefix = ' ' * indent
    return textwrap.fill(str(text), width=W - indent,
                         initial_indent=prefix, subsequent_indent=prefix)


# ── Candidate file helpers ────────────────────────────────────────────────────

from learn import CAND_FILE, miss_report, distill

def load_candidates(status_filter: str = 'pending') -> list:
    if not os.path.exists(CAND_FILE):
        return []
    results = []
    with open(CAND_FILE, encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
                r['_line'] = i
                if r.get('status') == status_filter:
                    results.append(r)
            except Exception:
                pass
    return results


def rewrite_candidates(updates: dict):
    """Update candidate statuses in-place by line index."""
    if not os.path.exists(CAND_FILE):
        return
    lines = []
    with open(CAND_FILE, encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
                if i in updates:
                    r.update(updates[i])
                lines.append(json.dumps(r))
            except Exception:
                lines.append(line.rstrip())
    with open(CAND_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ── Save approved cell ────────────────────────────────────────────────────────

def save_cell(bridge, description: str, content: str, source_table: str) -> str:
    cid = bridge.save_cell(
        description=description,
        content=content,
        source_table=source_table,
        confidence=0.90,
        energy=140.0,
    )
    return cid


# ── Review a single item ──────────────────────────────────────────────────────

def review_item(bridge, item: dict, index: int, total: int) -> str:
    """
    Show one candidate. Return action: 'approve', 'edit', 'skip', 'discard'.
    """
    hr()
    print(f"\n  {BOLD}Item {index} of {total}{RESET}  "
          f"{DIM}sim={item.get('avg_sim', '?')}  "
          f"source={item.get('source_table', '?')}{RESET}\n")

    print(f"  {CYAN}Description (what prospect asks):{RESET}")
    print(wrap(item.get('description', ''), indent=4))
    print()
    print(f"  {CYAN}Content (what Tee says):{RESET}")
    print(wrap(item.get('content', ''), indent=4))
    print()

    print(f"  {GREEN}a{RESET} approve   "
          f"{YELLOW}e{RESET} edit   "
          f"{DIM}s{RESET} skip   "
          f"{RED}d{RESET} discard\n")

    while True:
        raw = input("  Action: ").strip().lower()
        if raw in ('a', 'e', 's', 'd'):
            return raw
        print("  Enter a, e, s, or d")


def do_edit(item: dict) -> dict:
    """Let user edit description and/or content. Returns updated item."""
    print(f"\n  Current description: {item['description']!r}")
    new_desc = input("  New description (Enter to keep): ").strip()
    if new_desc:
        item['description'] = new_desc

    print(f"\n  Current content:\n")
    print(wrap(item['content'], indent=4))
    print("\n  New content (Enter on blank line to finish, Enter alone to keep):\n")
    lines = []
    while True:
        line = input("    ")
        if line == '':
            break
        lines.append(line)
    if lines:
        item['content'] = ' '.join(lines).strip()

    return item


# ── Review candidates queue ───────────────────────────────────────────────────

def review_candidates(bridge):
    candidates = load_candidates('pending')
    if not candidates:
        print(f"\n  {DIM}No pending candidates. Run with --distill to generate some.{RESET}\n")
        return

    print(f"\n  {len(candidates)} candidates from live sessions.\n")
    updates  = {}
    approved = 0
    skipped  = 0
    discarded= 0

    for i, item in enumerate(candidates, 1):
        action = review_item(bridge, item, i, len(candidates))

        if action == 'a':
            cid = save_cell(bridge, item['description'], item['content'], item['source_table'])
            print(f"\n  {GREEN}Saved: {cid[:42]}{RESET}")
            updates[item['_line']] = {'status': 'approved', 'approved_ts': time.time()}
            approved += 1

        elif action == 'e':
            item = do_edit(item)
            cid = save_cell(bridge, item['description'], item['content'], item['source_table'])
            print(f"\n  {GREEN}Saved (edited): {cid[:42]}{RESET}")
            updates[item['_line']] = {'status': 'approved', 'approved_ts': time.time(),
                                      'description': item['description'],
                                      'content': item['content']}
            approved += 1

        elif action == 's':
            skipped += 1

        elif action == 'd':
            updates[item['_line']] = {'status': 'discarded'}
            discarded += 1

    if updates:
        rewrite_candidates(updates)

    hr()
    print(f"\n  Candidates reviewed: {len(candidates)}")
    print(f"  {GREEN}Approved: {approved}{RESET}  |  Skipped: {skipped}  |  {RED}Discarded: {discarded}{RESET}\n")


# ── Review miss log ───────────────────────────────────────────────────────────

def review_misses(bridge):
    misses = miss_report(top_n=30)
    if not misses:
        print(f"\n  {DIM}No misses logged yet. Miss log is empty.{RESET}\n")
        return

    print(f"\n  Top {len(misses)} unanswered questions from miss log.\n")
    print("  For each one, write the cell content Tee should say, or skip.\n")

    approved = 0

    for i, miss in enumerate(misses, 1):
        hr()
        print(f"\n  {BOLD}Miss {i} of {len(misses)}{RESET}  "
              f"{DIM}asked {miss['count']}x  top_sim={miss['top_sim']:.3f}  "
              f"product={miss.get('product', 'none')}{RESET}\n")
        print(f"  {CYAN}Question:{RESET} {miss['query']!r}\n")

        print(f"  {GREEN}w{RESET} write a cell   "
              f"{DIM}s{RESET} skip   "
              f"{RED}d{RESET} dismiss (not a real gap)\n")

        while True:
            raw = input("  Action: ").strip().lower()
            if raw in ('w', 's', 'd'):
                break
            print("  Enter w, s, or d")

        if raw == 'w':
            print(f"\n  Write what Tee should say (blank line to finish):\n")
            lines = []
            while True:
                line = input("    ")
                if line == '':
                    break
                lines.append(line)
            content = ' '.join(lines).strip()
            if not content:
                print("  (Empty — skipped)")
                continue

            # Determine source table
            product = miss.get('product')
            if product:
                source_table = f'product_{product}'
            else:
                print("\n  Source table: reasoning / identity / conversation")
                source_table = input("  Enter type [reasoning]: ").strip() or 'reasoning'

            cid = save_cell(bridge, miss['query'], content, source_table)
            print(f"\n  {GREEN}Saved: {cid[:42]}{RESET}")
            approved += 1

        elif raw == 'd':
            print(f"  {DIM}Dismissed.{RESET}")

    hr()
    print(f"\n  Misses reviewed: {len(misses)}")
    print(f"  {GREEN}New cells written: {approved}{RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', action='store_true',
                        help='Review session candidates only')
    parser.add_argument('--misses',     action='store_true',
                        help='Review miss log only')
    parser.add_argument('--distill',    action='store_true',
                        help='Run distiller before reviewing candidates')
    args = parser.parse_args()

    banner("REVIEW QUEUE — Approve Tee's learning candidates")

    print("[Loading Tee...]", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("ERROR: substrate failed to load.")
        sys.exit(1)
    print(f"[{len(bridge.substrate.methodology_cells)} cells online]\n")

    if args.distill:
        print("[Distilling session logs...]\n")
        new = distill()
        print(f"  {len(new)} new candidates extracted from sessions.\n")

    do_candidates = args.candidates or (not args.candidates and not args.misses)
    do_misses     = args.misses     or (not args.candidates and not args.misses)

    if do_candidates:
        banner("QUEUE 1 — Session Candidates")
        review_candidates(bridge)

    if do_misses:
        banner("QUEUE 2 — Miss Log")
        review_misses(bridge)

    print(f"  Total cells now: {len(bridge.substrate.methodology_cells)}\n")
    print('=' * W + '\n')


if __name__ == '__main__':
    main()
