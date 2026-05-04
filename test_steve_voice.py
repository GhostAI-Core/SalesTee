#!/usr/bin/env python3
"""
test_steve_voice.py — Tests Steve's encoder→decoder pipeline on in-domain queries.
Bypasses Ollama entirely. Pure Steve: encode query → cosine match → decode response.
Shows both the decoder-generated text and the raw cell match for comparison.
"""

import os, sys, time, textwrap
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ── Load Steve ────────────────────────────────────────────────────────────────
from datag_bridge import DataGBridge
print("[Test] Loading Steve…", flush=True)
bridge = DataGBridge.get()
if not bridge.ready:
    print("[Test] FAILED — Steve's substrate didn't load")
    sys.exit(1)
print(f"[Test] Steve online — {len(bridge.substrate.methodology_cells)} cells\n")


# ── In-domain test queries ────────────────────────────────────────────────────
# These are things Steve should actually know about
QUERIES = [
    # Identity
    "who are you",
    "who built you",
    "what is your purpose",
    "what version are you",
    "can you learn",
    # Reasoning
    "how do you approach a new codebase",
    "what makes code maintainable",
    "how do you debug a problem",
    "what do you do when you're stuck",
    "how do you make architecture decisions",
    # Code
    "show me a python class",
    "how do you read a file in python",
    "what is a decorator",
    # Meta / conversational
    "hi",
    "can you fly?",
]

W = 72
print("=" * W)
print("  STEVE VOICE TEST — Encoder → Substrate → Decoder")
print(f"  Queries: {len(QUERIES)}  |  Cells: {len(bridge.substrate.methodology_cells)}")
print("=" * W)

results = []

for i, query in enumerate(QUERIES, 1):
    print(f"\n{'─' * W}")
    print(f"  [{i:02d}] QUERY: \"{query}\"")
    print(f"{'─' * W}")

    # 1) field_read — decoder-prioritised path
    t0 = time.time()
    decoded = bridge.field_read(query, k=5)
    dec_ms = (time.time() - t0) * 1000

    # 2) raw cell match — what the cosine search actually found
    t0 = time.time()
    hits = bridge.top_cells(query, k=3)
    cos_ms = (time.time() - t0) * 1000

    # 3) pure decoder generation (encode query → decode from latent)
    gen_text = None
    gen_ms = 0
    if bridge._dec is not None and bridge._enc is not None:
        t0 = time.time()
        gen_text = bridge._generate(query, temperature=0.7, top_k=30, max_new=80)
        gen_ms = (time.time() - t0) * 1000

    # Display results
    print(f"\n  STEVE (field_read, {dec_ms:.0f}ms):")
    if decoded:
        print(textwrap.fill(f"    \"{decoded}\"", width=W, subsequent_indent="    "))
    else:
        print("    [empty / no match]")

    print(f"\n  DECODER (pure generate, {gen_ms:.0f}ms):")
    if gen_text and gen_text.strip():
        print(textwrap.fill(f"    \"{gen_text}\"", width=W, subsequent_indent="    "))
    else:
        print("    [empty output]")

    if hits:
        best_sim, best_cid, best_cell = hits[0]
        raw_content = (getattr(best_cell, 'content', '') or '')[:150].strip()
        source = getattr(best_cell, 'source_table', '?')
        print(f"\n  CLOSEST CELL ({cos_ms:.0f}ms, sim={best_sim:.3f}, {source}):")
        print(textwrap.fill(f"    \"{raw_content}\"", width=W, subsequent_indent="    "))
    else:
        print("\n  CLOSEST CELL: [no matches]")

    results.append({
        "query": query,
        "field_read": decoded or "",
        "decoded": gen_text or "",
        "cell_sim": hits[0][0] if hits else 0,
        "cell_source": getattr(hits[0][2], 'source_table', '?') if hits else "?",
    })

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n\n{'=' * W}")
print("  SUMMARY")
print(f"{'=' * W}")

decoder_empty = sum(1 for r in results if len(r["decoded"].strip()) < 5)
field_empty   = sum(1 for r in results if len(r["field_read"].strip()) < 5)
avg_sim       = sum(r["cell_sim"] for r in results) / len(results)

print(f"  Total queries     : {len(results)}")
print(f"  Decoder empty     : {decoder_empty}/{len(results)}")
print(f"  field_read empty  : {field_empty}/{len(results)}")
print(f"  Avg cosine sim    : {avg_sim:.3f}")
print(f"{'=' * W}")

# Verdict
if decoder_empty <= 2 and field_empty <= 2:
    print("  ✅ Steve's voice is working — decoder is generating text from latent space")
elif decoder_empty > len(results) // 2:
    print("  ⚠️  Decoder is mostly empty — generating blanks from latent vectors")
    print("     The encoder→decoder pipeline may need more training epochs")
else:
    print("  ⚡ Mixed results — decoder works on some queries but not all")

print(f"{'=' * W}\n")
