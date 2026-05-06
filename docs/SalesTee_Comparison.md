# SalesTee — Capability & Performance Comparison
*Generated: 2026-05-05*

---

## Architecture Class

| System | Type | Parameters | Memory |
|---|---|---|---|
| **SalesTee** | Retrieval over neuronal substrate | ~2M (encoder) | ~50 MB |
| GPT-4o | LLM + RAG optional | ~1.7T (est.) | 80GB+ VRAM |
| Claude Sonnet | LLM | ~70B (est.) | 40GB+ VRAM |
| Llama 3.1 8B | Local LLM | 8B | 8GB VRAM |
| Gemini Flash | LLM | unknown | cloud only |
| Voiceflow / Bland AI | Rule graph + LLM call | N/A | cloud only |

---

## Capability Matrix

| Capability | SalesTee | GPT-4o | Llama 3.1 8B | Voiceflow |
|---|---|---|---|---|
| Runs fully offline | ✅ | ❌ | ✅ | ❌ |
| No API cost per call | ✅ | ❌ | ✅ | ❌ |
| Voice in/out | ✅ | ✅ (paid) | ❌ bare | ❌ bare |
| Product namespace isolation | ✅ | ✅ via prompt | ⚠️ manual | ✅ |
| Trains on your exact data | ✅ native | ⚠️ fine-tune | ⚠️ fine-tune | ⚠️ knowledge base |
| Handles novel questions | ❌ | ✅ | ✅ | ❌ |
| Generates new answers | ❌ retrieval only | ✅ | ✅ | ❌ |
| Stays on-script guaranteed | ✅ | ❌ | ❌ | ✅ |
| CRM / calendar integration | ❌ not yet | ✅ via tools | ✅ via tools | ✅ |
| Multi-turn context memory | ⚠️ last 1 turn | ✅ full window | ✅ full window | ⚠️ limited |
| Learns from live calls | ❌ not yet | ❌ | ❌ | ❌ |

---

## Hardware Requirements

| System | Minimum to run | GPU needed |
|---|---|---|
| **SalesTee** | 4GB RAM, any CPU | No |
| **SalesTee + Whisper (voice)** | 4GB RAM, any CPU | No |
| Llama 3.1 8B (quantised) | 8GB RAM, 8GB VRAM | Yes (or very slow) |
| GPT-4o | Browser / API | Cloud |
| Voiceflow | Browser | Cloud |

---

## Where SalesTee Wins

- **Cost per call: zero** — no tokens, no API, no per-minute billing
- **Deterministic** — same question always hits the same cell; fully auditable
- **Brand-safe by design** — cannot hallucinate outside the cells you wrote
- **Runs on a $400 machine** — no GPU, no cloud dependency
- **Instant to retrain** — add a cell and it is live in the same session, no fine-tune cycle

---

## Where SalesTee Loses Right Now

- **Novel questions** — if a prospect asks something with no matching cell above the similarity threshold, it says "Let me think about that" and stops there
- **Multi-turn reasoning** — context window is one turn deep (last Tee response, 60 chars). A real LLM holds 10+ turns
- **No generation** — it retrieves, it does not compose. Cannot combine two cells to form a new answer
- **Thin corpus** — 265 cells at current count. A well-prompted GPT-4o with a product doc will outperform on coverage
- **Encoder trained on small corpus** — DNA vectors are reasonable but not as discriminative as a model trained on millions of examples

---

## Honest Position

SalesTee today is a **deterministic, zero-cost, fully offline sales retrieval agent** with voice.
It is closer to a very smart FAQ engine than to GPT-4o.

The architecture supports live learning, cell growth, and multi-product routing — but the corpus is still thin.

**The gap to close is coverage.** More cells, a better ingest pipeline, and graceful fallback for unanswered questions. Once the corpus reaches approximately 1,000 quality cells per product, the determinism and zero-cost advantages start to genuinely compete with LLM-based agents for structured sales conversations.

---

## Roadmap to Close the Gap

| Priority | Item | Effort |
|---|---|---|
| 1 | Fix ingest pipeline — ask user for description per chunk | Low |
| 2 | Graceful fallback for low-similarity queries | Low |
| 3 | Grow VOXI corpus to 200+ cells | Medium |
| 4 | Multi-turn context window (last 3 turns) | Medium |
| 5 | API wrapper for website / VOXI integration | Medium |
| 6 | Live call learning — save unanswered queries as new cells | High |
| 7 | Retrain encoder every 500 new cells | Medium |

---

*SalesTee is built on the DataG neuronal architecture. All retrieval is cosine similarity over 128-dim DNA vectors produced by a custom transformer encoder (TrainingTeeEncoder). No external model dependencies at inference time.*
