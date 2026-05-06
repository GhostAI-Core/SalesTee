# SalesTee — Action Plan: Closing the Gaps
*Generated: 2026-05-05 | Current corpus: 269 cells*

---

## The Core Problem to Solve

SalesTee is a retrieval engine. Its quality ceiling is determined by three things in order:

1. **Corpus quality and coverage** — what cells exist and how well they are described
2. **Encoder discrimination** — how accurately DNA vectors separate different topics
3. **Runtime intelligence** — what the system does when retrieval fails or is ambiguous

Everything below flows from that hierarchy. Fix the corpus first. Then the encoder. Then the runtime. In that order.

---

## Phase 1 — Corpus Foundation (Now → 500 cells per product)
*Goal: no prospect question goes unanswered*

### 1.1 Fix the Ingest Pipeline (ingest_doc.py)

**Current problem:** Description is auto-generated from the first sentence of each chunk. This produces weak, generic DNA vectors. The description IS the DNA — it determines what questions hit the cell.

**Fix:** Ask the user to write the description for each chunk. One line. The question a prospect would ask to need this answer.

Example prompt during ingest:
```
Chunk preview: "VOXI cross-references your calendar and books meetings..."
What question does this answer? > can voxi book meetings automatically
```

This single change is the highest-leverage improvement in the entire system.

**Implementation:** Update `ingest_doc.py` to pause per chunk and ask for a description. Add a `--batch` mode for power users who want to review all chunks first and label them in one pass.

### 1.2 Gap Detection (new: gap_finder.py)

**Problem:** We don't know what questions Tee can't answer until a prospect asks them.

**Fix:** Build a script that takes a list of prospect questions and reports:
- Which cells fire (and at what similarity)
- Which questions fall below the 0.20 threshold (dead zones)
- Which questions are pulling the wrong cell (sim > 0.20 but wrong content)

This gives a systematic view of corpus gaps instead of discovering them one by one in live sessions.

**Usage:** Run before every product launch. Feed it 50 likely prospect questions. Fix every dead zone before going live.

### 1.3 Cell Quality Standard

Every cell needs three things to be useful:
- **Description:** the exact question a prospect would ask (drives DNA)
- **Content:** a clean, direct answer in Tee's voice (what gets spoken)
- **Source table:** correct namespace (identity / reasoning / product_x)

Reject any cell that fails these three. The ingest pipeline should enforce this.

### 1.4 Corpus Targets

| Namespace | Now | Phase 1 target | Phase 2 target |
|---|---|---|---|
| meth_identity | 33 | 60 | 100 |
| meth_reasoning | 203 | 300 | 500 |
| meth_product_voxi | 33 | 150 | 300 |
| meth_product_[next] | 0 | 100 | 200 |

At 150 VOXI cells, coverage for a standard sales conversation is solid. At 300, edge cases are covered.

---

## Phase 2 — Runtime Intelligence (500 → 1,000 cells)
*Goal: Tee handles anything, even if imperfectly*

### 2.1 Graceful Fallback for Dead Zones

**Current behaviour:** similarity < 0.20 → "Let me think about that." Full stop. No recovery.

**Fix:** Three-tier response strategy:

```
Tier 1: sim >= 0.40  → fire the cell directly (high confidence)
Tier 2: sim 0.20–0.39 → fire the cell + add a clarifying question
         ("I want to make sure I'm answering the right thing — are you asking about X?")
Tier 3: sim < 0.20   → honest deflection + capture the question
         ("That's outside what I know right now — let me flag that for the team.")
         → save the unanswered query to a miss_log.jsonl file
```

The miss log becomes the next ingest queue. Every unanswered question is a cell that needs to be written.

### 2.2 Multi-Turn Context Window

**Current:** context enrichment uses only the last Tee response, truncated to 60 chars.

**Fix:** Build a proper context window — last 3 exchanges, weighted (most recent = highest weight). Use it to build the retrieval query for vague inputs.

```python
def build_query(user_msg, history, window=3):
    kind = _classify(user_msg)
    if kind == 'identity': return user_msg
    if kind == 'vague' and history:
        recent = history[-window:]
        ctx = ' '.join(h['tee'][:80] for h in recent)
        return f"{ctx} {user_msg}"
    return user_msg
```

This makes "tell me more", "how?", "really?" work correctly across multi-turn threads instead of only anchoring to the last line.

### 2.3 Clarifying Question Bank (new cell type: meth_clarify)

When Tee is in Tier 2 (uncertain), she needs good clarifying questions — not generic ones.

Build a small set of clarifying cells per product that Tee can fire when confidence is medium:
- "Are you asking about setup or pricing?"
- "Is this for inbound calls, outbound, or both?"
- "Are you currently using another system for this?"

These keep the conversation alive instead of stalling.

### 2.4 Live Miss Capture (auto-growth seed)

Every Tier 3 miss gets written to `data_store/misses/miss_log.jsonl`:

```json
{"timestamp": "...", "query": "can voxi integrate with salesforce", "top_sim": 0.11, "product_scope": "voxi"}
```

Weekly review of the miss log → write cells for the top 10 misses → run ingest. This is how the corpus grows from real conversations rather than guesswork.

---

## Phase 3 — Encoder Evolution (1,000+ cells)
*Goal: DNA vectors that actually discriminate at scale*

### 3.1 Scheduled Retraining Trigger

The current encoder was trained on ~265 cells. At 1,000 cells, it needs a retrain. At 5,000 cells, it needs a larger model.

**Rule:** Retrain when corpus grows by 50% since last training run.

| Corpus size | Encoder size | Expected training time (CPU) |
|---|---|---|
| < 500 cells | tiny.en (current: 128-dim, 4 layers) | 5–10 min |
| 500–2,000 cells | small (256-dim, 6 layers) | 20–40 min |
| 2,000–10,000 cells | medium (512-dim, 8 layers) | 2–4 hours |
| 10,000+ cells | large (1024-dim, 12 layers) or switch to sentence-transformers | GPU recommended |

`train_training_tee.py` already exists. Just run it after each major corpus batch. The DNA vectors in existing cells need to be re-encoded after a retrain — build a `reindex.py` script that re-encodes all cells with the new encoder and saves updated DNA without touching content.

### 3.2 Reindex Script (new: reindex_substrate.py)

After every encoder retrain:
1. Load new encoder
2. For every cell in data_store/methodology/, re-encode the description → new DNA
3. Write updated JSON back to disk
4. Hot-reload into substrate

This keeps DNA current with the encoder without touching cell content or metadata.

### 3.3 Contrastive Pair Mining

As the corpus grows, InfoNCE training needs harder negatives — cells that are semantically close but should be kept separate. Build a script that:
1. Finds cell pairs with cosine similarity 0.70–0.90 (close but distinct)
2. Flags them as hard negative pairs
3. Includes them in the next training run with extra penalty weight

This forces the encoder to pull apart similar-sounding but different cells — critical when you have 200 VOXI cells and they start blending together.

---

## Phase 4 — API & Integration Layer
*Goal: Tee runs on the website, not just the terminal*

### 4.1 FastAPI Wrapper (new: api_tee.py)

Expose Tee's retrieval engine as a local REST API:

```
POST /chat
  body: { "message": "...", "session_id": "...", "product": "voxi" }
  response: { "reply": "...", "sim": 0.87, "cell_id": "..." }

POST /voice
  body: multipart audio file
  response: { "transcript": "...", "reply": "...", "audio_url": "..." }

GET /products
  response: ["voxi", "product_b"]

GET /health
  response: { "cells": 269, "ready": true }
```

Session state (history, used_cells, product_scope) lives server-side keyed by session_id. The website sends a session_id cookie; Tee maintains context across page refreshes.

### 4.2 WebSocket for Voice (real-time)

For the web avatar / live voice integration:

```
WS /stream
  client sends: audio chunks (16kHz PCM)
  server sends: { "transcript": "...", "reply": "...", "tts_audio": base64 }
```

Whisper runs on the server. TTS audio is streamed back chunk by chunk so the avatar starts speaking before the full sentence is generated.

### 4.3 Docker Container

Single container: Python + Tee substrate + Whisper + edge-tts. No external dependencies.

```dockerfile
FROM python:3.11-slim
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "api_tee.py"]
```

Mount `data_store/methodology/` as a volume so the corpus can grow without rebuilding the image. Mount `models/` as a volume so the encoder can be swapped without a rebuild.

---

## Phase 5 — Live Learning
*Goal: Tee gets smarter from every conversation*

### 5.1 Conversation Distillation

After each session, a background job:
1. Reads the session transcript
2. Identifies turns where Tee fired a high-confidence cell (sim > 0.70) AND the conversation continued naturally (no "what?" or "that doesn't answer my question")
3. Marks those as validated pairs — description + content that worked
4. Optionally surfaces them for human review before committing

These become the training signal for the next encoder retrain. The corpus learns from what actually worked in real conversations.

### 5.2 Human Review Queue (new: review_queue.py)

A lightweight admin tool — not a full UI, just a CLI — that shows:
- Miss log entries (unanswered questions → write a cell?)
- Low-confidence fires (sim 0.20–0.40 → is this the right cell?)
- Candidate cells from conversation distillation → approve/reject/edit

This is the human-in-the-loop that keeps quality high as the corpus scales.

### 5.3 A/B Cell Testing

When two cells compete for the same question space, track which one produces better conversation continuity. Keep the winner. This requires session logging — which the miss log infrastructure already supports.

---

## Infrastructure Principles (the 100-step view)

These govern every decision as the system scales:

**1. The corpus IS the product.**
Every feature, encoder improvement, and API wrapper is worthless if the cells are thin or badly described. Corpus quality always comes first.

**2. Description drives everything.**
The description field determines DNA. DNA determines retrieval. A bad description means a cell that never fires. Enforce description quality at the point of ingest — not after.

**3. Namespaces scale horizontally.**
Adding a new product is: write cells with `source_table=meth_product_[name]`. Nothing else changes. The routing layer already handles N products. This is the right design — don't break it.

**4. JSON files on disk scale to ~50,000 cells.**
At that point, migrate to SQLite or DuckDB for the substrate. The LivingCell schema stays identical — it's just the storage backend that changes. Design the bridge so the storage layer is swappable without touching chat_tee.py or api_tee.py.

**5. The encoder is a component, not the system.**
When the corpus outgrows the custom encoder, swap in `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 6 layers, 80MB). The bridge's `encode()` method is the only thing that changes. Everything else is the same.

**6. Voice is an interface, not a feature.**
talk_tee.py and the WebSocket API are just different input/output wrappers around the same retrieval engine. Keep them thin. The intelligence lives in the substrate.

**7. Every miss is a future cell.**
Never discard unanswered queries. They are the corpus roadmap written by real prospects.

---

## Build Order

```
Now         Fix ingest_doc.py — description per chunk
Week 1      gap_finder.py — systematic dead zone detection
Week 1      Grow VOXI corpus to 150 cells using gap finder output
Week 2      Three-tier similarity response (Tier 1/2/3 + miss log)
Week 2      Multi-turn context window (3 turns)
Week 3      FastAPI wrapper — api_tee.py
Week 3      reindex_substrate.py — re-encode all cells after encoder retrain
Week 4      Docker container + volume mounts
Month 2     WebSocket voice streaming for web avatar
Month 2     Review queue CLI for miss log + low-confidence fires
Month 3     Retrain encoder on 500+ cell corpus (small model)
Month 3+    Conversation distillation + A/B cell testing
```

---

## What We Are Not Building

- **A general LLM.** SalesTee is a specialist. It knows the products it is given and the sales craft it has been trained on. It does not need to know about the weather.
- **A prompt-engineered wrapper around GPT.** That is a different product with different economics. SalesTee's value is zero marginal cost per call and full auditability.
- **A UI before the engine works.** The CLI and voice interface are sufficient until the corpus is solid. Build the engine first.

---

*The goal is a system where adding a new product takes an afternoon, not a sprint. Where every unanswered question automatically becomes a work item. Where the corpus grows from real conversations. That is the architecture we are building toward.*
