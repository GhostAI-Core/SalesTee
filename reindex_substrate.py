#!/usr/bin/env python3
"""
reindex_substrate.py — Re-encode all cell DNA vectors with the current encoder.

Run this after every encoder retrain. Reads every cell in data_store/methodology/,
re-encodes its description field, writes the updated DNA back to disk.
Cell content, metadata, and all other fields are untouched.

Usage:
    python reindex_substrate.py
    python reindex_substrate.py --dry-run   # show what would change, save nothing
    python reindex_substrate.py --verbose   # print every cell as it is processed
"""

import os
import sys
import json
import argparse
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

_ROOT    = os.path.dirname(os.path.abspath(__file__))
METH_DIR = os.path.join(_ROOT, 'data_store', 'methodology')


def reindex(dry_run: bool = False, verbose: bool = False):
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("ERROR: substrate failed to load — run train_training_tee.py first")
        sys.exit(1)

    files = [f for f in os.listdir(METH_DIR) if f.endswith('.json')]
    print(f"\n[Reindex] {len(files)} cells found")
    print(f"[Reindex] dry_run={dry_run}\n")

    updated  = 0
    skipped  = 0
    errors   = 0
    t_start  = time.time()

    for i, fname in enumerate(sorted(files), 1):
        path = os.path.join(METH_DIR, fname)
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)

            description = (data.get('meta') or {}).get('description', '').strip()
            if not description:
                if verbose:
                    print(f"  [{i:4d}] SKIP (no description): {fname}")
                skipped += 1
                continue

            new_dna = bridge.encode(description).tolist()
            data['dna'] = new_dna

            if not dry_run:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)

            if verbose:
                print(f"  [{i:4d}] OK  {description[:60]!r}")
            updated += 1

        except Exception as e:
            print(f"  [{i:4d}] ERROR {fname}: {e}")
            errors += 1

        if i % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  ... {i}/{len(files)} processed ({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\n[Reindex] Complete in {elapsed:.1f}s")
    print(f"  Updated : {updated}")
    print(f"  Skipped : {skipped}  (no description field)")
    print(f"  Errors  : {errors}")

    if dry_run:
        print("\n  (dry run — no files written)")
    else:
        print(f"\n  All {updated} cells re-encoded with current encoder.")
        print("  Restart chat_tee.py or talk_tee.py to load updated DNA.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',  action='store_true', help='Preview only, do not write')
    parser.add_argument('--verbose',  action='store_true', help='Print every cell processed')
    args = parser.parse_args()
    reindex(dry_run=args.dry_run, verbose=args.verbose)
