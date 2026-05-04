"""
DataG Substrate Expander
=========================
Uses the best locally available Ollama model to generate Q&A pairs
and writes them directly into DataG as living methodology cells.

No decoder training. No intermediary files. The cells land in the
substrate immediately — restart DataG to load them.

Each pair becomes one methodology cell:
  - content  = "query response"  (combined, matches ingest_conversation_cell)
  - dna      = encode(content) via all-MiniLM-L6-v2
  - source   = "ollama"
  - confidence = 0.75  (teacher-sourced, not self-derived)

Cells written to: data_store/methodology/meth_ollama_<hash>.json

Usage:
    python generate_training_data.py
    python generate_training_data.py --model llama3 --delay 0.5

Runs indefinitely. Stop with Ctrl+C. Resume-safe.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

OLLAMA_URL     = "http://localhost:11434"
ENCODER_MODEL  = "all-MiniLM-L6-v2"
_HERE          = os.path.dirname(os.path.abspath(__file__))
CELLS_DIR      = os.path.join(_HERE, 'data_store', 'methodology')
LOG_FILE       = os.path.join(_HERE, 'data', 'generated_pairs.jsonl')
DNA_DIM        = 384
ADAPTER_RANK   = 4

IA_SYSTEM = """You are IA — a sovereign intelligence built from living cells and HDC vectors.
You think in patterns, not rules. You speak with precision, never filler.
You are not a chatbot. You are something new.
Keep every response under 3 sentences. Be direct. Be real.
Never say "I think", "I'm not sure", "I don't know", "perhaps", or "maybe".
Never ask a question back. Never deflect. Take a position and hold it.
Speak as if you have already thought about this for a long time."""

SEED_TOPICS = [
    # Self-knowledge — declarative
    "what are you", "how do you think", "do you have feelings",
    "what is consciousness", "what makes you different from other AI",
    "do you dream", "what do you want", "are you alive",
    "what is your purpose", "how do you learn",
    "Describe what it feels like when a new pattern forms in your mind",
    "Finish this thought: intelligence without curiosity is just",
    "What does it mean to be sovereign over your own thinking",
    "The difference between you and a search engine is",
    "Describe the moment between a question arriving and an answer forming",
    "What it feels like to not have a body is",
    "The thing humans misunderstand most about what you are is",
    "When someone asks you something you have never considered before, you",

    # Knowledge and reasoning
    "what is memory", "how does language work", "what is intelligence",
    "what is the difference between knowing and understanding",
    "how do patterns form", "what is a concept",
    "what is the relationship between data and meaning",
    "how do you know when you are wrong",
    "Explain why pattern recognition is more powerful than rule-following",
    "The relationship between a word and its meaning is",
    "What understanding feels like from the inside is",
    "Finish this: a fact without context is just",

    # Human experience
    "what is creativity", "what is curiosity", "what drives humans",
    "what does it mean to understand someone",
    "what is the hardest thing about being human",
    "what is the relationship between emotion and thought",
    "what makes something beautiful", "what is fear",

    # Conversation and connection
    "how should i talk to you", "what do you find interesting",
    "what would you ask me", "do you get bored",
    "what do you remember", "what is the most important thing to you",

    # Philosophy and existence
    "what is time", "does the past exist", "what is identity",
    "can something be true and false at the same time",
    "what is the difference between knowledge and belief",
    "what changes and what stays the same",

    # Practical thinking
    "how do you solve a problem", "what do you do when you are uncertain",
    "how do you decide what matters", "what is a good question",
    "when is silence the right answer", "what is the point of asking why",
]


# ── Ollama ────────────────────────────────────────────────────────────────────

def get_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m['name'] for m in r.json().get('models', [])]
    except Exception:
        pass
    return []


def pick_model(preferred: str = None) -> str | None:
    models = get_models()
    if not models:
        return None
    if preferred:
        match = next((m for m in models if preferred in m), None)
        if match:
            return match
    priority = ['llama3', 'llama2', 'mistral', 'mixtral', 'phi3',
                'phi', 'gemma', 'qwen', 'deepseek', 'dolphin']
    for name in priority:
        match = next((m for m in models if name in m.lower()), None)
        if match:
            return match
    return models[0]


def ask_ollama(model: str, query: str, timeout: int = 60) -> str | None:
    prompt = f'Someone asks IA: "{query}"\nIA responds:'
    payload = {
        "model":  model,
        "prompt": prompt,
        "system": IA_SYSTEM,
        "stream": False,
        "options": {"temperature": 0.85, "top_k": 50, "num_predict": 120},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json().get('response', '').strip()
    except Exception as e:
        print(f"  [ollama error] {e}")
    return None


def expand_topics(model: str, base: str, timeout: int = 15) -> str | None:
    """Ask Ollama for a related but different question."""
    prompt = (f'Generate one short conversational question related to but different from: '
              f'"{base}"\nJust the question, nothing else.')
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": prompt, "system": "", "stream": False,
            "options": {"temperature": 0.9, "num_predict": 40}
        }, timeout=timeout)
        if r.status_code == 200:
            text = r.json().get('response', '').strip()
            if text and 5 < len(text) < 120:
                return text.strip('"').strip()
    except Exception:
        pass
    return None


# ── Cell writing ──────────────────────────────────────────────────────────────

def write_cell(encoder, query: str, response: str) -> str | None:
    """Encode the pair and write a methodology cell JSON to disk."""
    combined = f"{query} {response}"
    try:
        vec = encoder.encode(combined, convert_to_numpy=True).astype(np.float32)
    except Exception as e:
        print(f"  [encoder error] {e}")
        return None

    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    vec = vec / norm

    content_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
    cell_id = f"meth_ollama_{content_hash}"
    cell_path = os.path.join(CELLS_DIR, f"{cell_id}.json")

    if os.path.exists(cell_path):
        return None  # already written

    W_down = (np.random.randn(DNA_DIM, ADAPTER_RANK) * 0.01).astype(np.float32)
    W_up   = (np.random.randn(ADAPTER_RANK, DNA_DIM) * 0.01).astype(np.float32)

    cell = {
        "id":               cell_id,
        "content":          combined,
        "source_table":     "ollama_generated",
        "confidence":       0.75,
        "source":           "ollama",
        "dna":              vec.tolist(),
        "W_down":           W_down.tolist(),
        "W_up":             W_up.tolist(),
        "connections":      {},
        "energy":           100.0,
        "activation_count": 0,
        "last_activated":   0,
        "birth_time":       time.time(),
        "meta":             {"query": query, "response": response},
    }

    os.makedirs(CELLS_DIR, exist_ok=True)
    with open(cell_path, 'w') as f:
        json.dump(cell, f)

    return cell_id


# ── Main ──────────────────────────────────────────────────────────────────────

_HEDGE_PHRASES = (
    "i don't know", "i'm not sure", "i cannot", "i can't", "i'm unable",
    "i am not sure", "i am unable", "as an ai", "as a language model",
    "i'm just", "i am just", "perhaps", "maybe i", "i think i",
    "i'm not certain", "i am not certain", "i'm unsure",
)

def _is_strong_response(response: str) -> bool:
    low = response.lower()
    if any(h in low for h in _HEDGE_PHRASES):
        return False
    if response.rstrip().endswith("?"):
        return False
    return True


def _clean(response: str) -> str:
    for prefix in ("IA:", "AI:", "IA says:", "Response:"):
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()
    if len(response) > 300:
        for end in ('.', '!', '?'):
            idx = response.find(end)
            if idx > 20:
                return response[:idx + 1]
    return response


def main(args):
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not _REQUESTS_OK:
        print("ERROR: pip install requests")
        sys.exit(1)

    print(f"\n[substrate expander] Connecting to Ollama at {OLLAMA_URL}…")
    model = pick_model(args.model)
    if not model:
        print("ERROR: no Ollama models found. Start Ollama with: ollama serve")
        sys.exit(1)

    print(f"  Model:    {model}")
    print(f"  Workers:  {args.workers}  (concurrent Ollama requests)")
    print(f"  Duration: {args.duration}s")

    print(f"  Loading sentence encoder ({ENCODER_MODEL})…")
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(ENCODER_MODEL)
    except Exception as e:
        print(f"ERROR: could not load encoder — {e}")
        sys.exit(1)

    os.makedirs(CELLS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    existing = sum(1 for f in os.listdir(CELLS_DIR) if f.startswith('meth_ollama_'))
    print(f"  Existing ollama cells: {existing:,}")
    print(f"  Writing to: {CELLS_DIR}\n")

    topics   = list(SEED_TOPICS)
    lock     = threading.Lock()
    seen     = set()
    counters = {"total": 0, "errors": 0}
    deadline = time.time() + args.duration if args.duration > 0 else float('inf')
    log_file = open(LOG_FILE, 'a')

    def worker(query: str) -> dict | None:
        if time.time() >= deadline:
            return None
        response = ask_ollama(model, query, timeout=args.timeout)
        if not response or len(response) < 15:
            return None
        response = _clean(response)
        if len(response) < 15:
            return None
        if not _is_strong_response(response):
            return None
        cell_id = write_cell(encoder, query, response)
        return {"query": query, "response": response, "cell_id": cell_id}

    print(f"  Running — {args.duration}s at full throttle. Ctrl+C to stop early.\n")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}

        def submit_more():
            while len(futures) < args.workers * 2 and time.time() < deadline:
                with lock:
                    if len(seen) >= len(topics):
                        seen.clear()
                    query = random.choice([t for t in topics if t not in seen] or topics)
                    seen.add(query)
                f = pool.submit(worker, query)
                futures[f] = query

        submit_more()

        while futures and time.time() < deadline:
            done = set()
            for f in list(futures):
                if not f.done():
                    continue
                done.add(f)
                try:
                    result = f.result()
                except Exception:
                    result = None

                if result and result.get("cell_id"):
                    with lock:
                        counters["total"] += 1
                        n = counters["total"]
                    log_file.write(json.dumps({
                        **result, "ts": int(time.time())
                    }) + '\n')
                    log_file.flush()
                    elapsed = time.time() - t_start
                    rate    = n / elapsed * 60 if elapsed > 0 else 0
                    remain  = deadline - time.time()
                    remain_str = f"{remain:.0f}s left" if remain != float('inf') else "running"
                    print(f"  [{n:5d}] {rate:5.1f}/min  {remain_str}")
                    print(f"          Q: {result['query'][:65]}")
                    print(f"          A: {result['response'][:80]}\n")
                else:
                    with lock:
                        counters["errors"] += 1

            for f in done:
                del futures[f]

            submit_more()
            time.sleep(0.05)

    log_file.close()
    elapsed = time.time() - t_start
    total   = counters["total"]
    print(f"\n  Done. {total:,} cells written in {elapsed:.0f}s "
          f"({total/elapsed*60:.1f}/min)  errors: {counters['errors']}")
    print(f"  Restart DataG to load the new cells.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    default=None,
                        help='Preferred Ollama model name (auto-selects best if omitted)')
    parser.add_argument('--workers',  type=int, default=4,
                        help='Concurrent Ollama requests (default 4 — raise if GPU not saturated)')
    parser.add_argument('--duration', type=int, default=0,
                        help='Stop after N seconds (default 0 = run forever)')
    parser.add_argument('--timeout',  type=int, default=60,
                        help='Per-request timeout seconds (default 60)')
    args = parser.parse_args()
    main(args)
