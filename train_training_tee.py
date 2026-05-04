#!/usr/bin/env python3
"""
train_training_tee.py — Train Training Tee's custom encoder and decoder.

Run from Training Tee directory with DataG venv active:
    python train_training_tee.py

Outputs:
    models/tokenizer.json
    models/encoder.pt
    models/decoder.pt
"""

import os
import sys
import json
import math
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'neural'))
from tokenizer import TrainingTeeTokenizer
from encoder import TrainingTeeEncoder
from decoder_net import TrainingTeeDecoder

METH_DIR  = os.path.join(os.path.dirname(__file__), 'data_store', 'methodology')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 800
BATCH_SIZE    = 8
LR            = 3e-4
WARMUP_STEPS  = 100
MAX_LEN       = 128
D_MODEL       = 128
OUT_DIM       = 128
NHEAD         = 4
NUM_LAYERS    = 4
DIM_FF        = 256
DROPOUT       = 0.1
TEMP_CONT     = 0.05   # InfoNCE temperature — lower = tighter clusters
WEIGHT_CONT   = 0.8    # contrastive loss weight — higher = stronger clustering
WEIGHT_DEC    = 0.2    # decoder CE loss weight
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED          = 42


# ── Data ──────────────────────────────────────────────────────────────────────

def load_pairs() -> list[tuple[str, str]]:
    """Load (description, content) pairs from seed cells."""
    pairs = []
    for fname in os.listdir(METH_DIR):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(METH_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            meta    = data.get('meta', {})
            desc    = meta.get('description', '').strip()
            content = data.get('content', '').strip()
            if desc and content and len(content) >= 8:
                pairs.append((desc, content))
        except Exception:
            pass
    return pairs


def augment(text: str) -> str:
    """Light augmentation: random word drop or shuffle prefix."""
    words = text.split()
    if len(words) <= 3:
        return text
    r = random.random()
    if r < 0.15:
        # Drop one random word
        drop = random.randint(0, len(words) - 1)
        words = words[:drop] + words[drop + 1:]
    elif r < 0.25:
        # Swap two adjacent words
        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
    return ' '.join(words)


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tok: TrainingTeeTokenizer,
                 max_len: int = MAX_LEN, augment_queries: bool = True):
        self.pairs    = pairs
        self.tok      = tok
        self.max_len  = max_len
        self.augment  = augment_queries

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        desc, content = self.pairs[idx]
        if self.augment:
            desc = augment(desc)
        q_ids = self.tok.encode(desc,    max_len=self.max_len)
        c_ids = self.tok.encode(content, max_len=self.max_len)
        return q_ids, c_ids


def collate(batch, pad_id: int):
    q_batch, c_batch = zip(*batch)

    def pad(seqs):
        max_l = max(len(s) for s in seqs)
        ids   = torch.zeros(len(seqs), max_l, dtype=torch.long)
        mask  = torch.ones(len(seqs),  max_l, dtype=torch.bool)   # True = PAD
        for i, s in enumerate(seqs):
            ids[i, :len(s)]  = torch.tensor(s)
            mask[i, :len(s)] = False
        return ids, mask

    q_ids, q_mask = pad(q_batch)
    c_ids, c_mask = pad(c_batch)
    return q_ids, q_mask, c_ids, c_mask


# ── Loss ──────────────────────────────────────────────────────────────────────

def infonce_loss(q_vecs: torch.Tensor, c_vecs: torch.Tensor,
                 temperature: float = TEMP_CONT) -> torch.Tensor:
    """
    InfoNCE (NT-Xent) contrastive loss.
    q_vecs, c_vecs: (B, D) L2-normalized.
    Diagonal = positive pairs.
    """
    sim = torch.matmul(q_vecs, c_vecs.T) / temperature  # (B, B)
    labels = torch.arange(len(q_vecs), device=q_vecs.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


def decoder_loss(encoder: TrainingTeeEncoder, decoder: TrainingTeeDecoder,
                 c_ids: torch.Tensor, c_mask: torch.Tensor,
                 latent: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Teacher-forced decoder loss: given latent, predict content tokens.
    tgt_in  = content[:-1]  (BOS ... last-1)
    tgt_out = content[1:]   (1 ... EOS)
    """
    tgt_in  = c_ids[:, :-1]
    tgt_out = c_ids[:, 1:]
    tgt_pad = c_mask[:, :-1]

    logits = decoder(latent, tgt_in, tgt_pad_mask=tgt_pad)  # (B, T-1, V)
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        tgt_out.reshape(B * T),
        ignore_index=pad_id,
    )
    return loss


# ── Training ──────────────────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup: int, total: int):
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading cell pairs…")
    pairs = load_pairs()
    print(f"  {len(pairs)} pairs found")
    if not pairs:
        print("No pairs. Run seed_training_tee.py first.")
        return

    print("Building tokenizer…")
    all_texts = [t for pair in pairs for t in pair]
    tok = TrainingTeeTokenizer().build(all_texts, min_freq=1)
    tok.save(os.path.join(MODEL_DIR, 'tokenizer.json'))
    print(f"  Vocab size: {tok.vocab_size}")

    print("Building encoder + decoder…")
    encoder = TrainingTeeEncoder(
        vocab_size=tok.vocab_size, d_model=D_MODEL, out_dim=OUT_DIM,
        nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT,
    ).to(DEVICE)
    decoder = TrainingTeeDecoder(
        vocab_size=tok.vocab_size, latent_dim=OUT_DIM, d_model=D_MODEL,
        nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"  Encoder params: {encoder.param_count():,}")
    print(f"  Decoder params: {decoder.param_count():,}")

    dataset = PairDataset(pairs, tok, max_len=MAX_LEN)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_id),
    )

    params    = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    total_steps = EPOCHS * len(loader)
    scheduler = make_scheduler(optimizer, WARMUP_STEPS, total_steps)

    print(f"\nTraining on {DEVICE} for {EPOCHS} epochs…\n")
    best_loss  = float('inf')
    step       = 0
    t0         = time.time()

    for epoch in range(1, EPOCHS + 1):
        encoder.train(); decoder.train()
        epoch_loss = 0.0
        n_batches  = 0

        for q_ids, q_mask, c_ids, c_mask in loader:
            q_ids  = q_ids.to(DEVICE);  q_mask = q_mask.to(DEVICE)
            c_ids  = c_ids.to(DEVICE);  c_mask = c_mask.to(DEVICE)

            q_vecs = encoder(q_ids, q_mask)   # (B, D)
            c_vecs = encoder(c_ids, c_mask)   # (B, D)

            l_cont = infonce_loss(q_vecs, c_vecs)
            l_dec  = decoder_loss(encoder, decoder, c_ids, c_mask, q_vecs, tok.pad_id)
            loss   = WEIGHT_CONT * l_cont + WEIGHT_DEC * l_dec

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/{EPOCHS}  loss={avg:.4f}  lr={lr_now:.2e}  [{elapsed:.0f}s]")

        if avg < best_loss:
            best_loss = avg
            torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder.pt'))
            torch.save(decoder.state_dict(), os.path.join(MODEL_DIR, 'decoder.pt'))

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Models saved to {MODEL_DIR}/")

    # Quick sanity check
    _sanity(tok, encoder, decoder)

    # Recompute DNA for all cells using the trained encoder
    _recompute_dna(tok, encoder)


def _sanity(tok, encoder, decoder):
    print("\n-- Sanity check ------------------------------------------------")
    encoder.eval(); decoder.eval()

    tests = [
        ("who are you",           ["tee", "salestee"]),
        ("who created you",       ["garth", "built"]),
        ("what do you do",        ["pipeline", "framework", "sales"]),
        ("what is a swimlane",    ["criteria", "target", "focus"]),
        ("what is your philosophy",["solve", "problem", "fluff"]),
        ("what is voxi",          ["voxi", "voice", "ai", "workforce"]),
        ("how do i set up voxi",  ["sign", "step", "voxi", "forwarding"]),
    ]

    for query, keywords in tests:
        ids    = tok.encode(query)
        q_tens = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            latent = encoder(q_tens).squeeze(0)
            gen    = decoder.generate(latent, tok.bos_id, tok.eos_id,
                                      max_new=30, temperature=0.7)
        text = tok.decode(gen)
        hit  = '+' if any(k in text.lower() for k in keywords) else '?'
        print(f"  {hit} Q: {query!r:30s} -> {text[:60]!r}")
    print()


def _recompute_dna(tok, encoder):
    """Recompute DNA vectors for ALL methodology cells using the trained encoder."""
    print("-- Recomputing DNA vectors --------------------------------------")
    encoder.eval()
    updated = 0

    for fname in os.listdir(METH_DIR):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(METH_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)

            desc = data.get('meta', {}).get('description', '').strip()
            if not desc:
                continue

            # Encode description using the custom encoder
            ids = tok.encode(desc, max_len=MAX_LEN)
            t_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                vec = encoder(t_ids).squeeze(0).cpu().numpy()

            data['dna'] = vec.tolist()

            with open(path, 'w') as f:
                json.dump(data, f)
            updated += 1
        except Exception as e:
            print(f"  WARN: {fname}: {e}")

    print(f"  Updated {updated} cells with custom encoder DNA (128-dim)")
    print()


if __name__ == '__main__':
    train()

