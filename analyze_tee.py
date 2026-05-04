#!/usr/bin/env python3
"""
analyze_tee.py — 10-Point Stability Analysis for SalesTee
═══════════════════════════════════════════════════════════

Runs a comprehensive diagnostic across encoder, decoder, and tokenizer
to verify that Training Tee is stable and ready for product ingestion.

Tests:
  1. Identity Recall          — Does Tee know who she is?
  2. Creator Attribution      — Does she know Garth / her origins?
  3. Sales Methodology Recall — Can she retrieve core sales concepts?
  4. Embedding Separation     — Are semantically different queries far apart?
  5. Embedding Clustering     — Are similar queries close together?
  6. Decoder Coherence        — Does generated text make grammatical sense?
  7. Decoder Diversity        — Does sampling produce varied outputs?
  8. OOV / Edge-case Handling — Does she handle unknown inputs gracefully?
  9. Latent Space Norm Check  — Are encoder outputs properly normalized?
 10. Round-trip Fidelity      — Encode → Decode: does meaning survive?
"""

import os, sys, json, math, io
import torch
import torch.nn.functional as F

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'neural'))
from tokenizer import TrainingTeeTokenizer
from encoder   import TrainingTeeEncoder
from decoder_net import TrainingTeeDecoder

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Architecture constants (must match training) ─────────────────────────────
D_MODEL    = 128
OUT_DIM    = 128
NHEAD      = 4
NUM_LAYERS = 4
DIM_FF     = 256
DROPOUT    = 0.0   # inference — no dropout


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_models():
    tok = TrainingTeeTokenizer.load(os.path.join(MODEL_DIR, 'tokenizer.json'))

    encoder = TrainingTeeEncoder(
        vocab_size=tok.vocab_size, d_model=D_MODEL, out_dim=OUT_DIM,
        nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'encoder.pt'),
                                        map_location=DEVICE, weights_only=True))
    encoder.eval()

    decoder = TrainingTeeDecoder(
        vocab_size=tok.vocab_size, latent_dim=OUT_DIM, d_model=D_MODEL,
        nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT,
    ).to(DEVICE)
    decoder.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'decoder.pt'),
                                        map_location=DEVICE, weights_only=True))
    decoder.eval()
    return tok, encoder, decoder


def encode(tok, encoder, text: str) -> torch.Tensor:
    ids = tok.encode(text)
    t   = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return encoder(t).squeeze(0)  # (D,)


def decode(tok, decoder, latent: torch.Tensor,
           max_new: int = 50, temperature: float = 0.7) -> str:
    with torch.no_grad():
        ids = decoder.generate(latent, tok.bos_id, tok.eos_id,
                               max_new=max_new, temperature=temperature)
    return tok.decode(ids)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


# ── Separator ─────────────────────────────────────────────────────────────────

SEP = "-" * 70
PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

score_card: list[tuple[str, str, str]] = []


def report(test_num: int, name: str, status: str, detail: str):
    score_card.append((f"T{test_num:02d}", name, status))
    tag = PASS if status == "PASS" else (WARN if status == "WARN" else FAIL)
    print(f"\n{SEP}")
    print(f"  T{test_num:02d} │ {name}")
    print(f"  {tag}")
    print(f"  {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def t01_identity_recall(tok, enc, dec):
    """Does Tee know who she is?"""
    queries = ["who are you", "what is your name", "tell me about yourself"]
    hits = 0
    outputs = []
    keywords = ["tee", "salestee", "sales", "training"]
    for q in queries:
        lat = encode(tok, enc, q)
        out = decode(tok, dec, lat).lower()
        outputs.append(f"    Q: {q!r:35s} → {out[:80]!r}")
        if any(k in out for k in keywords):
            hits += 1
    detail = "\n".join(outputs)
    detail += f"\n    Keyword hits: {hits}/{len(queries)}"
    status = "PASS" if hits >= 2 else ("WARN" if hits >= 1 else "FAIL")
    report(1, "Identity Recall", status, detail)


def t02_creator_attribution(tok, enc, dec):
    """Does she know Garth / origins?"""
    queries = ["who created you", "who is your creator", "where does garth live"]
    hits = 0
    outputs = []
    keywords = ["garth", "roodepoort", "south africa", "creator", "built"]
    for q in queries:
        lat = encode(tok, enc, q)
        out = decode(tok, dec, lat).lower()
        outputs.append(f"    Q: {q!r:35s} → {out[:80]!r}")
        if any(k in out for k in keywords):
            hits += 1
    detail = "\n".join(outputs)
    detail += f"\n    Keyword hits: {hits}/{len(queries)}"
    status = "PASS" if hits >= 2 else ("WARN" if hits >= 1 else "FAIL")
    report(2, "Creator Attribution", status, detail)


def t03_sales_methodology(tok, enc, dec):
    """Can she retrieve core sales concepts?"""
    probes = [
        ("what is a swimlane",     ["criteria", "target", "geography", "industry", "size"]),
        ("what is bucketing",      ["technique", "categoriz", "leads", "status", "prospecting"]),
        ("what is a crm",          ["customer", "relationship", "management", "database", "foundation"]),
        ("what is outreach",       ["outreach", "salesloft", "acceleration", "tools", "email"]),
        ("what is a buyer persona",["description", "role", "needs", "concerns", "company"]),
    ]
    hits = 0
    outputs = []
    for q, kws in probes:
        lat = encode(tok, enc, q)
        out = decode(tok, dec, lat).lower()
        outputs.append(f"    Q: {q!r:35s} → {out[:80]!r}")
        if any(k in out for k in kws):
            hits += 1
    detail = "\n".join(outputs)
    detail += f"\n    Concept recall: {hits}/{len(probes)}"
    status = "PASS" if hits >= 4 else ("WARN" if hits >= 2 else "FAIL")
    report(3, "Sales Methodology Recall", status, detail)


def t04_embedding_separation(tok, enc, dec):
    """Are semantically different queries far apart in latent space?"""
    pairs = [
        ("who are you",          "what is bucket 4"),
        ("what is a crm",        "do you have feelings"),
        ("how to handle a lead", "where does garth live"),
    ]
    sims = []
    outputs = []
    for a, b in pairs:
        va = encode(tok, enc, a)
        vb = encode(tok, enc, b)
        s  = cosine_sim(va, vb)
        sims.append(s)
        outputs.append(f"    cos({a!r:30s}, {b!r:30s}) = {s:.4f}")
    avg = sum(sims) / len(sims)
    outputs.append(f"    Average separation: {avg:.4f}  (want < 0.60)")
    detail = "\n".join(outputs)
    status = "PASS" if avg < 0.60 else ("WARN" if avg < 0.75 else "FAIL")
    report(4, "Embedding Separation", status, detail)


def t05_embedding_clustering(tok, enc, dec):
    """Are similar queries close together?"""
    clusters = [
        (["who are you", "what is your name", "tell me about yourself"],        "identity"),
        (["what is a swimlane", "how do i define my swimlane"],                 "swimlane"),
        (["what is outreach", "what is the goal of a cold email"],              "outreach"),
    ]
    intra_sims = []
    outputs = []
    for texts, label in clusters:
        vecs = [encode(tok, enc, t) for t in texts]
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(cosine_sim(vecs[i], vecs[j]))
        avg = sum(sims) / len(sims) if sims else 0
        intra_sims.append(avg)
        outputs.append(f"    Cluster '{label}': avg intra-sim = {avg:.4f}")
    overall = sum(intra_sims) / len(intra_sims)
    outputs.append(f"    Overall intra-cluster similarity: {overall:.4f}  (want > 0.15)")
    detail = "\n".join(outputs)
    status = "PASS" if overall > 0.15 else ("WARN" if overall > 0.08 else "FAIL")
    report(5, "Embedding Clustering", status, detail)


def t06_decoder_coherence(tok, enc, dec):
    """Does generated text look like real sentences?"""
    queries = [
        "what do you do",
        "what is your philosophy",
        "how do i personalize a pitch",
    ]
    coherent = 0
    outputs = []
    for q in queries:
        lat = encode(tok, enc, q)
        out = decode(tok, dec, lat, max_new=40, temperature=0.5)
        words = out.split()
        # Heuristic: coherent if > 3 words, no excessive repetition
        unique_ratio = len(set(words)) / max(len(words), 1)
        is_ok = len(words) >= 3 and unique_ratio > 0.35
        tag = "✓" if is_ok else "✗"
        outputs.append(f"    {tag} Q: {q!r:35s} → {out[:80]!r}  (words={len(words)}, uniq={unique_ratio:.2f})")
        if is_ok:
            coherent += 1
    detail = "\n".join(outputs)
    detail += f"\n    Coherent: {coherent}/{len(queries)}"
    status = "PASS" if coherent == len(queries) else ("WARN" if coherent >= 2 else "FAIL")
    report(6, "Decoder Coherence", status, detail)


def t07_decoder_diversity(tok, enc, dec):
    """Does sampling produce varied outputs for the same query?"""
    query = "what is your purpose"
    lat = encode(tok, enc, query)
    generations = set()
    for _ in range(5):
        out = decode(tok, dec, lat, temperature=0.9)
        generations.add(out.strip().lower())
    unique = len(generations)
    detail = f"    Query: {query!r}\n"
    for i, g in enumerate(generations, 1):
        detail += f"    Run {i}: {g[:70]!r}\n"
    detail += f"    Unique outputs: {unique}/5  (want >= 2)"
    status = "PASS" if unique >= 2 else ("WARN" if unique >= 1 else "FAIL")
    report(7, "Decoder Diversity", status, detail)


def t08_oov_edge_cases(tok, enc, dec):
    """Does she handle unknown / OOV inputs gracefully (no crash, no empty)?"""
    edge_inputs = [
        "",                        # empty
        "xyzzy foo blargh",        # gibberish
        "a",                       # single char
        "!!!???...",               # punctuation only
        "what " * 50,              # excessive repetition
    ]
    survived = 0
    outputs = []
    for inp in edge_inputs:
        try:
            lat = encode(tok, enc, inp if inp else "empty")
            out = decode(tok, dec, lat, max_new=20)
            tag = "✓" if len(out.strip()) > 0 else "✗"
            outputs.append(f"    {tag} Input: {inp[:30]!r:35s} → {out[:60]!r}")
            if len(out.strip()) > 0:
                survived += 1
        except Exception as e:
            outputs.append(f"    ✗ Input: {inp[:30]!r:35s} → CRASHED: {e}")
    detail = "\n".join(outputs)
    detail += f"\n    Survived: {survived}/{len(edge_inputs)}"
    status = "PASS" if survived == len(edge_inputs) else ("WARN" if survived >= 3 else "FAIL")
    report(8, "OOV / Edge-case Handling", status, detail)


def t09_latent_norm_check(tok, enc, dec):
    """Are encoder outputs properly L2-normalized?"""
    queries = ["who are you", "what is a crm", "how to sell", "what is bucketing",
               "tell me about yourself", "do you have feelings"]
    norms = []
    for q in queries:
        v = encode(tok, enc, q)
        norms.append(float(v.norm()))
    avg_norm = sum(norms) / len(norms)
    max_dev  = max(abs(n - 1.0) for n in norms)
    detail  = f"    Norms: {[f'{n:.6f}' for n in norms]}\n"
    detail += f"    Avg norm: {avg_norm:.6f}  (want = 1.0000)\n"
    detail += f"    Max deviation: {max_dev:.6f}  (want < 0.001)"
    status = "PASS" if max_dev < 0.001 else ("WARN" if max_dev < 0.01 else "FAIL")
    report(9, "Latent Space Norm Check", status, detail)


def t10_roundtrip_fidelity(tok, enc, dec):
    """Encode a known description → Decode: does meaning survive?"""
    probes = [
        ("what is salestee",    ["sales", "tee", "agent", "tool", "ai"]),
        ("what is a swimlane",  ["lane", "focus", "vertical", "territory", "niche"]),
        ("what are the best sales tools", ["crm", "tool", "lead", "source", "contact"]),
    ]
    hits = 0
    outputs = []
    for q, kws in probes:
        lat = encode(tok, enc, q)
        out = decode(tok, dec, lat, temperature=0.5).lower()
        match = any(k in out for k in kws)
        tag = "✓" if match else "✗"
        outputs.append(f"    {tag} {q!r:40s} → {out[:80]!r}")
        if match:
            hits += 1
    detail = "\n".join(outputs)
    detail += f"\n    Semantic fidelity: {hits}/{len(probes)}"
    status = "PASS" if hits >= 2 else ("WARN" if hits >= 1 else "FAIL")
    report(10, "Round-trip Fidelity", status, detail)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SalesTee — 10-Point Stability Analysis")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Models: {MODEL_DIR}")

    tok, enc, dec = load_models()
    print(f"  Vocab:  {tok.vocab_size}")
    print(f"  Encoder params: {enc.param_count():,}")
    print(f"  Decoder params: {dec.param_count():,}")

    t01_identity_recall(tok, enc, dec)
    t02_creator_attribution(tok, enc, dec)
    t03_sales_methodology(tok, enc, dec)
    t04_embedding_separation(tok, enc, dec)
    t05_embedding_clustering(tok, enc, dec)
    t06_decoder_coherence(tok, enc, dec)
    t07_decoder_diversity(tok, enc, dec)
    t08_oov_edge_cases(tok, enc, dec)
    t09_latent_norm_check(tok, enc, dec)
    t10_roundtrip_fidelity(tok, enc, dec)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SCORECARD")
    print(f"{'=' * 70}")
    passes = sum(1 for _, _, s in score_card if s == "PASS")
    warns  = sum(1 for _, _, s in score_card if s == "WARN")
    fails  = sum(1 for _, _, s in score_card if s == "FAIL")
    for tid, name, status in score_card:
        tag = PASS if status == "PASS" else (WARN if status == "WARN" else FAIL)
        print(f"  {tid} │ {tag:10s} │ {name}")
    print(f"{'─' * 70}")
    print(f"  Result: {passes} PASS  /  {warns} WARN  /  {fails} FAIL")
    if fails == 0 and warns <= 2:
        print(f"\n  🟢  Tee is STABLE — ready for product data ingestion.")
    elif fails <= 2:
        print(f"\n  🟡  Tee is MARGINAL — review warnings before loading products.")
    else:
        print(f"\n  🔴  Tee needs RETRAINING — too many failures.")
    print()


if __name__ == '__main__':
    main()
