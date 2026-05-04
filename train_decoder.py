"""
IA Decoder Training Script
===========================
Trains the native decoder on (sentence_embedding, text) pairs.

The decoder learns one thing: given a 384-dim semantic vector,
produce the words that express it. Content lives in the substrate.
This script teaches IA how to speak.

Input formats (auto-detected):
  - Plain text (.txt):  one sentence per line
  - JSONL (.jsonl):     {"text": "..."} or {"response": "..."}

Usage:
    python train_decoder.py data/training.txt
    python train_decoder.py data/pairs.jsonl --epochs 20 --batch 32

After training, models/ia_decoder.pt and models/ia_tokenizer.json
are written. IAGenerator will load them automatically on next start.
"""

import argparse
import json
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from neural.ia_tokenizer import IATokenizer
from neural.ia_decoder   import IADecoder

MODELS_DIR   = os.path.join(_HERE, 'models')
ENCODER_MODEL = "all-MiniLM-L6-v2"
DNA_DIM       = 384
MAX_SEQ_LEN   = 64   # max tokens per sentence


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sentences(path: str) -> list[str]:
    sentences = []
    ext = os.path.splitext(path)[1].lower()

    if ext == '.jsonl':
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get('text') or obj.get('response') or obj.get('output', '')
                    if text and len(text.strip()) > 10:
                        sentences.append(text.strip())
                except json.JSONDecodeError:
                    continue
    else:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 10:
                    sentences.append(line)

    print(f"  Loaded {len(sentences):,} sentences from {os.path.basename(path)}")
    return sentences


def load_from_substrate(cells_dir: str) -> list[tuple[np.ndarray, str]]:
    """
    Load (dna_vector, content) pairs directly from methodology cell JSON files.
    Skips re-encoding — cells already have their DNA from all-MiniLM-L6-v2.
    This is the natural training set: the substrate teaches the decoder its own voice.
    """
    pairs = []
    skipped = 0
    for fname in os.listdir(cells_dir):
        if not fname.endswith('.json'):
            continue
        try:
            with open(os.path.join(cells_dir, fname)) as f:
                data = json.load(f)
            dna  = data.get('dna', [])
            if len(dna) != DNA_DIM:
                skipped += 1
                continue
            # Use response-only text when available (Ollama cells store it in meta)
            # This teaches the decoder to output answers, not "query answer" pairs
            meta     = data.get('meta', {})
            response = meta.get('response', '').strip() if meta else ''
            text     = response if len(response) >= 10 else data.get('content', '').strip()
            if not text or len(text) < 10:
                skipped += 1
                continue
            vec = np.array(dna, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                skipped += 1
                continue
            pairs.append((vec / norm, text))
        except Exception:
            skipped += 1
    print(f"  Substrate: {len(pairs):,} cells loaded ({skipped} skipped)")
    return pairs


# ── Dataset ───────────────────────────────────────────────────────────────────

class DecoderDataset(Dataset):
    """
    Built from pre-encoded (dna_vector, content) pairs.
    Accepts output from either load_sentences+encoder or load_from_substrate.
    """

    def __init__(self, pairs: list[tuple[np.ndarray, str]], tokenizer: IATokenizer,
                 max_len: int = MAX_SEQ_LEN):
        self.pairs = []
        for vec, text in pairs:
            try:
                ids = tokenizer.encode(text)
                if len(ids) > max_len + 2:
                    ids = ids[:max_len + 1] + [tokenizer.eos_id]
                self.pairs.append((vec, ids))
            except Exception:
                continue
        print(f"  Dataset: {len(self.pairs):,} valid pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch, pad_id: int, max_len: int):
    vecs, seqs = zip(*batch)
    max_seq = min(max(len(s) for s in seqs), max_len + 2)
    padded  = []
    for s in seqs:
        s = s[:max_seq]
        padded.append(s + [pad_id] * (max_seq - len(s)))
    vecs_t = torch.tensor(np.stack(vecs), dtype=torch.float32)  # (B, 384)
    ids_t  = torch.tensor(padded, dtype=torch.long)             # (B, T)
    return vecs_t, ids_t


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[train_decoder] device={device}")

    # Tokenizer — always initialise first
    print("  Initialising tokenizer…")
    tokenizer = IATokenizer()
    os.makedirs(MODELS_DIR, exist_ok=True)
    tokenizer.save(os.path.join(MODELS_DIR, 'ia_tokenizer.json'))

    # Data — two paths
    if args.from_substrate:
        cells_dir = os.path.join(_HERE, 'data_store', 'methodology')
        if not os.path.isdir(cells_dir):
            print(f"ERROR: substrate cells directory not found: {cells_dir}")
            sys.exit(1)
        raw_pairs = load_from_substrate(cells_dir)
        if not raw_pairs:
            print("ERROR: no cells found in substrate. Run generate_training_data.py first.")
            sys.exit(1)
        random.shuffle(raw_pairs)
        dataset = DecoderDataset(raw_pairs, tokenizer)
    else:
        if not args.data:
            print("ERROR: provide a data file (--data path) or use --from-substrate")
            sys.exit(1)
        print(f"  Loading sentence encoder ({ENCODER_MODEL})…")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(ENCODER_MODEL)
        sentences = load_sentences(args.data)
        if not sentences:
            print("ERROR: no sentences loaded. Check your data file.")
            sys.exit(1)
        random.shuffle(sentences)
        raw_pairs = [(encoder.encode(s, convert_to_numpy=True).astype('float32'), s)
                     for s in sentences]
        raw_pairs = [(v / (np.linalg.norm(v) + 1e-8), t) for v, t in raw_pairs
                     if np.linalg.norm(v) > 1e-8]
        dataset = DecoderDataset(raw_pairs, tokenizer)

    if len(dataset) < 10:
        print("ERROR: too few valid pairs to train.")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_id, MAX_SEQ_LEN),
        num_workers=0,
    )

    # Model
    model = IADecoder(
        vocab_size=tokenizer.vocab_size,
        dna_dim=DNA_DIM,
        model_dim=384,
        n_heads=6,
        n_layers=4,
        ffn_dim=1024,
        max_len=MAX_SEQ_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params/1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print(f"\n  Training {args.epochs} epochs × {len(loader)} batches…\n")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (vecs, ids) in enumerate(loader):
            vecs = vecs.to(device)           # (B, 384)
            ids  = ids.to(device)            # (B, T)

            # Context: sentence embedding as the single cross-attention slot
            ctx = vecs.unsqueeze(1)          # (B, 1, 384)

            # Teacher forcing: input = ids[:, :-1], target = ids[:, 1:]
            inp    = ids[:, :-1]
            target = ids[:, 1:]

            pad_mask = (inp == tokenizer.pad_id)

            logits = model(inp, ctx, pad_mask=pad_mask)  # (B, T-1, vocab)
            loss   = criterion(logits.reshape(-1, tokenizer.vocab_size),
                               target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        if avg < best_loss:
            best_loss = avg
            decoder_path = os.path.join(MODELS_DIR, 'ia_decoder.pt')
            model.save(decoder_path)
            print(f"    ✓ Saved → {decoder_path}")

    print(f"\n[train_decoder] Done. Best loss: {best_loss:.4f}")
    print(f"  Restart IA — IAGenerator will load the trained decoder automatically.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the IA decoder to speak from substrate DNA vectors.'
    )
    parser.add_argument('--data',            default=None,
                        help='Path to training data (.txt or .jsonl) — one sentence per line')
    parser.add_argument('--from-substrate',  action='store_true',
                        help='Train directly from substrate methodology cells (no encoder re-run)')
    parser.add_argument('--epochs',  type=int,   default=30,   help='Training epochs (default 30)')
    parser.add_argument('--batch',   type=int,   default=16,   help='Batch size (default 16)')
    parser.add_argument('--lr',      type=float, default=3e-4, help='Learning rate (default 3e-4)')
    args = parser.parse_args()

    if not args.from_substrate and not args.data:
        parser.print_help()
        print("\nEither --from-substrate or --data <file> is required.")
        sys.exit(1)

    if args.data and not os.path.exists(args.data):
        print(f"ERROR: file not found: {args.data}")
        sys.exit(1)

    train(args)
