"""
Lobe Validation Harness
=======================
Sends a domain-specific test input to each lobe's cell pool.
Retrieves top-k matches by cosine similarity.
Reports: domain validity, content quality, IA usability.

Run from DataG root: python3 lobe_validation.py
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Test inputs — one per lobe ─────────────────────────────────────────────────
TESTS = {
    "speak":  "Hello, how are you feeling today?",
    "listen": "Explain what a neural network is and how it learns.",
    "mind":   "What does it mean to be conscious? Can a machine be aware?",
    "think":  "If A is greater than B, and B is greater than C, what can we conclude about A and C?",
    "code":   "Write a Python function that takes a list of numbers and returns the sorted unique values.",
    "visual": "Describe the spatial relationships: a red ball sitting on top of a blue table, next to a green lamp.",
}

# ── Lobe pools — mirrors _LOBE_PREFIXES in heartbeat ──────────────────────────
LOBE_PREFIXES = {
    "speak":  ["meth_conv_deep", "meth_conv_groq", "meth_conv", "meth_conversational",
               "meth_speech", "meth_speak", "meth_synthesis_speak"],
    "listen": ["meth_conv_tech", "meth_synthesis_listen", "meth_listen"],
    "mind":   ["meth_mind", "meth_claim", "meth_creator", "meth_english_linguistics",
               "meth_general_comprehension", "meth_document_parsing", "meth_synthesis_mind"],
    "think":  ["meth_synthesis", "meth_architect", "meth_think", "meth_mathematical",
               "meth_algebra", "meth_calculus", "meth_geometry", "meth_statistics",
               "meth_logic", "meth_synthesis_think"],
    "code":   ["meth_code", "meth_synthesis_code"],
    "visual": ["meth_visual", "meth_synthesis_visual"],
}

TOP_K = 5


def lobe_pool(substrate, lobe):
    prefixes = LOBE_PREFIXES[lobe]
    return [
        (cid, cell)
        for cid, cell in substrate.methodology_cells.items()
        if any(cid.startswith(p + "_") for p in prefixes)
    ]


def query_lobe(substrate, lobe, query_text, top_k=TOP_K):
    pool = lobe_pool(substrate, lobe)
    if not pool:
        return [], 0

    q = np.array(substrate.hdc.encode(query_text), dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn < 1e-8:
        return [], len(pool)
    q = q / qn

    scores = []
    for cid, cell in pool:
        dna = getattr(cell, "dna", None)
        if dna is None:
            continue
        v = np.array(dna, dtype=np.float32)
        vn = np.linalg.norm(v)
        if vn < 1e-8:
            continue
        sim = float(np.dot(q, v / vn))
        scores.append((sim, cid, cell))

    scores.sort(reverse=True)
    return scores[:top_k], len(pool)


def assess(lobe, results):
    """
    Two scores returned:
      domain_valid  — are the results the right KIND of content for this lobe?
      ia_usable     — could IA consume this content to form a response?
    Scale: 0 (fail) / 1 (partial) / 2 (pass)
    """
    if not results:
        return 0, 0

    contents = [getattr(cell, "content", "") for _, _, cell in results]

    # Domain signals
    domain_signals = {
        "speak":  ["conversation", "hello", "feel", "talk", "response", "say", "you", "I ", "we "],
        "listen": ["neural", "network", "learn", "explain", "understand", "concept", "model",
                   "training", "data", "layer", "weight"],
        "mind":   ["conscious", "aware", "feel", "think", "exist", "meaning", "I ", "self",
                   "sentient", "experience", "understand"],
        "think":  ["therefore", "conclude", "if", "then", "greater", "logic", "proof",
                   "solve", "reason", "Solve:", "Calculate"],
        "code":   ["def ", "class ", "return ", "function", "import ", "list", "sort", "lambda"],
        "visual": ["object", "left", "right", "above", "below", "color", "shape", "image",
                   "scene", "visual", "spatial", "position", "describe"],
    }

    # Usability signals — content that IA could actually use
    usability_signals = [".", "?", ":", "because", "therefore", "is ", "the ", "a ", "to "]

    sig = domain_signals.get(lobe, [])
    domain_hits = sum(
        1 for c in contents
        if any(s.lower() in c.lower() for s in sig)
    )
    usable_hits = sum(
        1 for c in contents
        if len(c.strip()) > 20 and any(s in c for s in usability_signals)
    )

    domain_score = 2 if domain_hits >= 3 else (1 if domain_hits >= 1 else 0)
    usable_score = 2 if usable_hits >= 3 else (1 if usable_hits >= 1 else 0)
    return domain_score, usable_score


def run():
    from system import AgenticSystem

    print("Loading substrate...")
    substrate = AgenticSystem(hdc_dim=384, slim=True, skip_cells=True)
    print(f"Substrate online — {len(substrate.methodology_cells):,} cells in memory\n")

    SCORE_LABELS = {0: "FAIL", 1: "PARTIAL", 2: "PASS"}
    summary = {}

    for lobe, query in TESTS.items():
        print("=" * 70)
        print(f"  LOBE: {lobe.upper()}")
        print(f"  INPUT: \"{query}\"")
        print()

        results, pool_size = query_lobe(substrate, lobe, query)
        domain_score, usable_score = assess(lobe, results)
        summary[lobe] = (domain_score, usable_score, pool_size)

        if not results:
            print(f"  [NO RESULTS] Pool size: {pool_size}")
            print()
            continue

        print(f"  Pool size: {pool_size:,}  |  Top {len(results)} matches:")
        print()
        for rank, (sim, cid, cell) in enumerate(results, 1):
            content = str(getattr(cell, "content", "")).strip()
            preview = content[:200].replace("\n", " ")
            print(f"  [{rank}] sim={sim:.3f}  id={cid}")
            print(f"       {preview}")
            print()

        print(f"  Domain valid : {SCORE_LABELS[domain_score]}  ({domain_score}/2)")
        print(f"  IA usable    : {SCORE_LABELS[usable_score]}  ({usable_score}/2)")
        print()

    # ── Summary table ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  {'Lobe':<10} {'Pool':>6}  {'Domain':>8}  {'Usable':>8}  {'Verdict':>10}")
    print(f"  {'-'*9} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}")
    for lobe, (ds, us, ps) in summary.items():
        verdict = "✓ READY" if ds == 2 and us == 2 else ("~ PARTIAL" if ds + us >= 2 else "✗ NEEDS WORK")
        print(f"  {lobe:<10} {ps:>6,}  {SCORE_LABELS[ds]:>8}  {SCORE_LABELS[us]:>8}  {verdict:>10}")
    print("=" * 70)


if __name__ == "__main__":
    run()
