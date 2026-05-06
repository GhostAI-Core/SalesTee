#!/usr/bin/env python3
"""
gap_finder.py — Find dead zones in Tee's corpus before they hit a prospect.

Reads a list of prospect questions (one per line) and reports:
  - GREEN  sim >= 0.50  : strong hit, correct cell
  - YELLOW sim 0.20-0.49: weak hit, check the cell content
  - RED    sim < 0.20   : dead zone — no cell covers this question

Usage:
    python gap_finder.py                        # interactive — type questions
    python gap_finder.py questions.txt          # batch file, one question per line
    python gap_finder.py --product voxi         # scope to one product namespace
    python gap_finder.py questions.txt --top 5  # show top 5 hits per question
"""

import os
import sys
import argparse
import textwrap

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

W = 76

GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
DIM    = '\033[90m'
BOLD   = '\033[1m'
RESET  = '\033[0m'


def colour(sim: float) -> str:
    if sim >= 0.50: return GREEN
    if sim >= 0.20: return YELLOW
    return RED


def label(sim: float) -> str:
    if sim >= 0.50: return 'STRONG'
    if sim >= 0.20: return 'WEAK  '
    return 'DEAD  '


def _in_scope(cell, product: str | None) -> bool:
    t = getattr(cell, 'source_table', '')
    if product is None:
        return True
    if not t.startswith('meth_product'):
        return True
    return t == f'meth_product_{product}'


def check_question(bridge, question: str, product: str | None, top_k: int) -> dict:
    hits = bridge.top_cells(question, k=top_k + 10)
    scoped = [(s, cid, c) for s, cid, c in hits if _in_scope(c, product)]
    if not scoped:
        return {'question': question, 'sim': 0.0, 'hits': []}

    top = scoped[:top_k]
    best_sim, best_cid, best_cell = scoped[0]
    hits_out = []
    for s, cid, c in top:
        desc    = (getattr(c, 'meta', {}) or {}).get('description', cid)
        content = (getattr(c, 'content', '') or '')[:80].strip()
        src     = getattr(c, 'source_table', '')
        hits_out.append({'sim': s, 'desc': desc, 'content': content, 'src': src})

    return {'question': question, 'sim': best_sim, 'hits': hits_out}


def print_result(result: dict, top_k: int, verbose: bool):
    q   = result['question']
    sim = result['sim']
    c   = colour(sim)
    lbl = label(sim)
    hits = result['hits']

    print(f"\n  {c}{BOLD}{lbl}{RESET}  {sim:.3f}  {q!r}")

    if verbose and hits:
        for i, h in enumerate(hits[:top_k], 1):
            dim = DIM if i > 1 else ''
            print(f"    {dim}#{i} sim={h['sim']:.3f}  [{h['src']}]  {h['desc']!r:.55}")
            print(f"         {h['content']!r:.70}{RESET}")


def run_batch(bridge, questions: list, product: str | None, top_k: int, verbose: bool):
    strong, weak, dead = [], [], []

    for q in questions:
        result = check_question(bridge, q, product, top_k)
        print_result(result, top_k, verbose)
        sim = result['sim']
        if sim >= 0.50:   strong.append(result)
        elif sim >= 0.20: weak.append(result)
        else:             dead.append(result)

    print(f"\n{'─'*W}")
    print(f"  {len(questions)} questions checked\n")
    print(f"  {GREEN}{BOLD}{len(strong):>3} STRONG{RESET}  sim >= 0.50  — good coverage")
    print(f"  {YELLOW}{BOLD}{len(weak):>3} WEAK  {RESET}  sim 0.20-0.49 — check cell content")
    print(f"  {RED}{BOLD}{len(dead):>3} DEAD  {RESET}  sim < 0.20   — need new cells")
    print(f"{'─'*W}\n")

    if dead:
        print(f"  {RED}Dead zones — write these cells next:{RESET}\n")
        for r in dead:
            print(f"    - {r['question']!r}")
        print()

    if weak:
        print(f"  {YELLOW}Weak hits — review the cell content:{RESET}\n")
        for r in weak:
            best = r['hits'][0] if r['hits'] else {}
            print(f"    - {r['question']!r}")
            if best:
                print(f"      hits: {best['desc']!r:.60}  (sim={best['sim']:.3f})")
        print()

    return dead, weak


def run_interactive(bridge, product: str | None, top_k: int, verbose: bool):
    print(f"\n  Type a prospect question. Press Enter to check.")
    print(f"  Type 'quit' to exit.\n")
    print(f"{'─'*W}")

    while True:
        try:
            raw = input("\n  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        if raw.lower() in ('quit', 'exit', 'q'):
            break
        result = check_question(bridge, raw, product, top_k)
        print_result(result, top_k, verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file',      nargs='?', default=None,
                        help='Text file of questions, one per line')
    parser.add_argument('--product', default=None,
                        help='Scope to a product namespace (e.g. voxi)')
    parser.add_argument('--top',     type=int, default=3,
                        help='Number of top hits to show per question (default 3)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all top hits, not just best')
    parser.add_argument('--save-dead', default=None, metavar='FILE',
                        help='Write dead zone questions to a file for later ingest')
    args = parser.parse_args()

    print(f"\n{'='*W}")
    print(f"  GAP FINDER — Tee's coverage checker")
    if args.product:
        print(f"  Scoped to product: {args.product}")
    print(f"{'='*W}\n")

    print("[Loading Tee...]", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("ERROR: substrate failed to load.")
        sys.exit(1)

    cells = bridge.substrate.methodology_cells
    print(f"[{len(cells)} cells online]\n")

    if args.file:
        if not os.path.exists(args.file):
            print(f"  File not found: {args.file}\n")
            sys.exit(1)
        with open(args.file, encoding='utf-8') as f:
            questions = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"  {len(questions)} questions loaded from {args.file}\n")
        print('─' * W)
        dead, weak = run_batch(bridge, questions, args.product,
                               args.top, args.verbose or True)
        if args.save_dead and dead:
            with open(args.save_dead, 'w', encoding='utf-8') as f:
                for r in dead:
                    f.write(r['question'] + '\n')
            print(f"  Dead zones saved to: {args.save_dead}\n")
    else:
        run_interactive(bridge, args.product, args.top, verbose=True)


if __name__ == '__main__':
    main()
