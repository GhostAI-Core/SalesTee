#!/usr/bin/env python3
"""
reencode_cells.py — Replace MiniLM (384-dim) DNA vectors with SteveEncoder (128-dim).

Run after train_steve.py:
    python reencode_cells.py
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from tokenizer import SteveTokenizer
from encoder import SteveEncoder

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
METH_DIR  = os.path.join(os.path.dirname(__file__), 'data_store', 'methodology')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    tok     = SteveTokenizer.load(os.path.join(MODEL_DIR, 'tokenizer.json'))
    encoder = SteveEncoder(vocab_size=tok.vocab_size, d_model=128, out_dim=128,
                            nhead=4, num_layers=4, dim_feedforward=256)
    encoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'encoder.pt'),
                                        map_location=DEVICE))
    encoder.to(DEVICE).eval()
    return tok, encoder


@torch.no_grad()
def encode(text: str, tok: SteveTokenizer, encoder: SteveEncoder) -> list:
    ids   = tok.encode(text, max_len=128)
    tens  = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    vec   = encoder(tens).squeeze(0).cpu().numpy()
    return vec.tolist()


def main():
    print("Loading SteveEncoder…")
    tok, encoder = load_model()

    files   = glob.glob(os.path.join(METH_DIR, 'meth_*.json'))
    updated = 0
    skipped = 0

    for path in files:
        try:
            with open(path) as f:
                cell = json.load(f)

            # Get description — used as the semantic anchor for DNA
            meta = cell.get('meta', {})
            desc = meta.get('description', '').strip()
            if not desc:
                # Fall back to content prefix as description
                content = cell.get('content', '')
                desc = ' '.join(content.split()[:12])

            if not desc:
                skipped += 1
                continue

            cell['dna'] = encode(desc, tok, encoder)
            with open(path, 'w') as f:
                json.dump(cell, f)
            updated += 1

        except Exception as e:
            print(f"  [!] {os.path.basename(path)}: {e}")
            skipped += 1

    print(f"Done. Updated: {updated}  Skipped: {skipped}")
    print(f"DNA dimension: 128  (was: 384 MiniLM)")


if __name__ == '__main__':
    main()
