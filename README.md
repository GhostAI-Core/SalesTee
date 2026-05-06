# SalesTee

A deterministic, zero-cost, offline sales retrieval agent with voice. Built on the DataG neuronal architecture.

Tee retrieves answers from a substrate of memory cells using cosine similarity over 384-dim DNA vectors (sentence-transformers). No LLM at inference time — every response is a real cell you wrote, spoken word-for-word.

---

## Quick start (Docker)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API + Swagger | http://localhost:8000/docs |
| Customer UI (VOXI) | http://localhost:5173 |
| Admin panel | http://localhost:5174 |

`data_store/` is mounted as a volume — cells learned via the admin panel persist on your machine across container restarts.

---

## Quick start (local)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Mac/Linux

# Terminal 1 — API
.venv\Scripts\python -m uvicorn backend:app --reload --port 8000

# Terminal 2 — customer UI
cd voxi_ui && npm install && npm run dev     # → http://localhost:5173

# Terminal 3 — admin panel
cd voxi_admin && npm install && npm run dev  # → http://localhost:5174
```

---

## Voice (local only)

```bash
# VOXI-specific voice agent (recommended)
.venv\Scripts\python talk_voxi.py

# Generic voice agent
.venv\Scripts\python talk_tee.py

# Options
--voice en-ZA-LeahNeural   # change TTS voice
--debug                    # show retrieval similarity per turn
```

Tee listens automatically via VAD (no button to press). Silence for 1.5s ends the turn. Silence for 10s triggers a check-in; another 10s ends the call.

---

## Architecture

```
Prospect utterance
       │
       ▼
  build_query()          — classify (identity / vague / normal), enrich with context
       │
       ▼
  encode()               — sentence-transformers all-MiniLM-L6-v2 → 384-dim vector
       │
       ▼
  top_cells()            — cosine similarity over all cells in substrate
       │
       ▼
  filter + scope         — product namespace isolation, internal cell exclusion
       │
       ▼
  three-tier response    — ≥0.40 direct · 0.20–0.39 + clarifier · <0.20 deflect + log miss
       │
       ▼
  raw cell content       — spoken or returned word-for-word
```

### Cell namespaces

| source_table | Purpose |
|---|---|
| `meth_identity` | Who Tee is — tone, values, persona |
| `meth_reasoning` | Objection handling, persuasion |
| `meth_product_voxi` | VOXI product knowledge |
| `meth_product_<name>` | Any additional product |
| `meth_conversation` | General / session-learned cells |

---

## Key files

| File | Purpose |
|---|---|
| `backend.py` | FastAPI — chat, transcribe, ingest, misses endpoints |
| `datag_bridge.py` | Singleton substrate bridge — encode, retrieve, save cells |
| `chat_tee.py` | Core retrieval engine — query classification, three-tier response |
| `talk_tee.py` | CLI voice agent — VAD mic, Whisper STT, edge-tts TTS |
| `talk_voxi.py` | VOXI-specific voice agent with branded opener |
| `learn.py` | Session logger + miss logger + distiller |
| `ingest_doc.py` | Bulk doc/PDF ingestion pipeline |
| `reindex_substrate.py` | Re-encode all cell DNA after encoder change |
| `src/living_cell.py` | LivingCell data structure |
| `models/` | encoder.pt · decoder.pt · tokenizer.json (fallback encoder) |
| `data_store/methodology/` | All memory cells (JSON) |
| `data_store/misses/` | Unanswered query log — feeds admin panel |
| `data_store/sessions/` | Conversation session logs |
| `voxi_ui/` | Customer-facing VOXI chat UI (React + Vite) |
| `voxi_admin/` | Admin learning panel — answer missed questions (React + Vite) |
| `ingestor_ui/` | Doc upload + bulk cell creation UI (React + Vite) |

---

## Adding a new product

```bash
# Ingest a document or PDF
.venv\Scripts\python ingest_doc.py --file product_brief.pdf --product myproduct

# Or use the ingestor UI
cd ingestor_ui && npm run dev   # → http://localhost:5175
```

After ingestion Tee auto-detects the new product and routes questions to it with no config.

---

## Learning from missed questions

1. Open the admin panel at http://localhost:5174
2. Missed questions (asked but unanswered) appear in the left column, ranked by frequency
3. Click a question, type Tee's answer, pick the namespace, click **Add to Tee**
4. The cell is live immediately — no restart needed

---

## Reindexing after corpus growth

Run this after adding 50%+ more cells (keeps similarity scores accurate):

```bash
.venv\Scripts\python reindex_substrate.py
```

---

## Hardware

| Setup | Minimum |
|---|---|
| API + chat (text only) | 4 GB RAM, any CPU, no GPU |
| + Whisper STT (voice) | 4 GB RAM — tiny.en model, ~75 MB |
| + edge-tts (TTS) | Internet connection required |
| Docker | Docker Desktop, 4 GB RAM |
