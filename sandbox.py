#!/usr/bin/env python3
"""
sandbox.py — Steve × Qwen interaction sandbox.

DeepSeek plays: junior TTS/STT developer transitioning from warehousing.
Steve responds: via his own substrate (encoder → cosine search → decoder).

Usage:
    python sandbox.py [--turns N] [--model qwen2.5-coder:7b]

Output:
    logs/sandbox_<timestamp>.json   — full structured log
    logs/sandbox_<timestamp>.txt    — readable transcript
"""

import os
import sys
import json
import time
import argparse
import datetime
import textwrap
import requests

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

OLLAMA_URL  = "http://localhost:11434/api/chat"
LOG_DIR     = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DEEPSEEK_SYSTEM = """
You are Marcus — a 34-year-old who spent 11 years working logistics and inventory control
in a large warehousing company. Six months ago you made the switch to software development,
specifically text-to-speech (TTS) and speech-to-text (STT) systems. You are self-taught,
enthusiastic but clearly still learning.

Your personality:
- You use warehousing analogies to explain things ("it's like a pick-and-pack system but for phonemes")
- You get excited about small wins
- You ask basic but sincere questions
- You sometimes confuse terminology (e.g., "tokens" vs "phonemes", "latency" vs "lag")
- You have real practical ideas from your logistics background (batching, queuing, throughput)
- You genuinely want to learn from Steve and share what little you know

Your goal in this conversation:
- Share something you've been working on or thinking about (TTS/STT related)
- Ask Steve for ideas, methods, or ways to improve
- React to Steve's responses naturally — build on them, ask follow-ups
- Keep each message to 3-5 sentences max, conversational tone

Do NOT break character. Do NOT be overly technical. You are Marcus, the former warehouse guy.
Start by introducing yourself briefly and sharing what you're currently stuck on.
""".strip()


# ── Steve interface ────────────────────────────────────────────────────────────

def load_steve():
    """Load Steve's bridge — returns bridge instance."""
    print("[Sandbox] Loading Steve…", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        raise RuntimeError("Steve's substrate failed to load")
    print(f"[Sandbox] Steve online — {len(bridge.substrate.methodology_cells)} cells", flush=True)
    return bridge


def steve_respond(bridge, message: str) -> str:
    """
    Route a message through Steve's actual architecture.
    Tries field_read (semantic lookup + decoder), falls back gracefully.
    """
    response = bridge.field_read(message, k=5)
    if not response or len(response.strip()) < 5:
        # Secondary: try top cell content directly
        hits = bridge.top_cells(message, k=3)
        for sim, cid, cell in hits:
            content = (getattr(cell, 'content', '') or '').strip()
            if len(content) > 10:
                response = content
                break
    if not response:
        response = "Run the code. See what breaks."
    return response.strip()


# ── DeepSeek interface ─────────────────────────────────────────────────────────

def deepseek_respond(history: list[dict], model: str = "qwen2.5-coder:7b",
                     timeout: int = 120) -> tuple[str, float]:
    """Call Ollama chat endpoint. Returns (response_text, latency_secs)."""
    payload = {
        "model":    model,
        "messages": history,
        "stream":   False,
        "options":  {"temperature": 0.8, "num_predict": 256},
    }
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data["message"]["content"].strip()
        latency = time.time() - t0
        return text, latency
    except Exception as e:
        return f"[DeepSeek error: {e}]", time.time() - t0


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from DeepSeek output."""
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(log: dict) -> str:
    turns       = log["turns"]
    total_turns = len(turns)
    steve_lens  = [len(t["steve"].split()) for t in turns]
    ds_lens     = [len(t["deepseek"].split()) for t in turns]
    avg_steve   = sum(steve_lens) / max(len(steve_lens), 1)
    avg_ds      = sum(ds_lens)    / max(len(ds_lens), 1)

    # Detect Steve failures (fallback response or very short)
    failures = [i+1 for i, t in enumerate(turns)
                if len(t["steve"].split()) < 4 or "run the code" in t["steve"].lower()]

    # Topics mentioned in DeepSeek's messages
    tts_stt_kw  = ["tts", "stt", "speech", "voice", "audio", "phoneme", "latency",
                   "model", "buffer", "chunk", "stream", "transcri", "wav", "sample",
                   "encode", "decode", "token", "pipeline", "batch", "queue", "throughput"]
    topics = set()
    for t in turns:
        low = t["deepseek"].lower()
        for kw in tts_stt_kw:
            if kw in low:
                topics.add(kw)

    lines = [
        "=" * 60,
        "  SANDBOX ANALYSIS",
        "=" * 60,
        f"  Turns completed  : {total_turns}",
        f"  Steve avg words  : {avg_steve:.1f}  (per response)",
        f"  Marcus avg words : {avg_ds:.1f}  (per message)",
        f"  Steve failures   : {len(failures)} {'(turns: ' + str(failures) + ')' if failures else '(none)'}",
        f"  Domain topics    : {', '.join(sorted(topics)) or 'none detected'}",
        "",
        "  STEVE RESPONSES",
        "  " + "-" * 40,
    ]
    for i, t in enumerate(turns):
        tag = "⚠" if (i+1) in failures else "✓"
        lines.append(f"  {tag} Turn {i+1:02d}: {t['steve'][:80]!r}")

    lines += [
        "",
        "  VERDICT",
        "  " + "-" * 40,
    ]
    if len(failures) == 0:
        lines.append("  Steve held the conversation without failures.")
    elif len(failures) <= total_turns // 3:
        lines.append("  Steve managed most turns. Substrate needs more cells in TTS/STT domain.")
    else:
        lines.append("  Steve struggled. Needs TTS/STT seed cells added to the substrate.")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=8,
                        help="Number of conversation turns (default: 8)")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    args = parser.parse_args()

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_json = os.path.join(LOG_DIR, f"sandbox_{ts}.json")
    log_txt  = os.path.join(LOG_DIR, f"sandbox_{ts}.txt")

    print(f"\n{'='*60}")
    print(f"  STEVE × QWEN SANDBOX")
    print(f"  Turns: {args.turns}  |  Model: {args.model}")
    print(f"{'='*60}\n")

    # Load Steve
    bridge = load_steve()

    # DeepSeek message history (system + conversation)
    ds_history = [{"role": "system", "content": DEEPSEEK_SYSTEM}]

    log = {
        "timestamp": ts,
        "model":     args.model,
        "turns":     [],
        "metadata":  {
            "steve_cells": len(bridge.substrate.methodology_cells),
            "planned_turns": args.turns,
        }
    }
    transcript_lines = [
        f"SANDBOX SESSION — {ts}",
        f"Steve cells: {len(bridge.substrate.methodology_cells)} | Model: {args.model}",
        "=" * 60,
        "",
    ]

    for turn_n in range(1, args.turns + 1):
        print(f"\n── Turn {turn_n}/{args.turns} " + "─" * 40)

        # ── DeepSeek speaks ───────────────────────────────────────────────
        print(f"  [Marcus] thinking…", end=" ", flush=True)
        ds_raw, ds_lat = deepseek_respond(ds_history, model=args.model)
        ds_clean       = strip_think(ds_raw)
        print(f"({ds_lat:.1f}s)")
        print(f"\n  Marcus: {textwrap.fill(ds_clean, width=70, subsequent_indent='          ')}\n")

        # Add to history
        ds_history.append({"role": "assistant", "content": ds_clean})

        # ── Steve responds ────────────────────────────────────────────────
        print(f"  [Steve]  searching substrate…", end=" ", flush=True)
        t0      = time.time()
        steve_r = steve_respond(bridge, ds_clean)
        s_lat   = time.time() - t0
        print(f"({s_lat*1000:.0f}ms)")
        print(f"\n  Steve:   {textwrap.fill(steve_r, width=70, subsequent_indent='          ')}\n")

        # Feed Steve's response back so DeepSeek can react to it
        ds_history.append({
            "role": "user",
            "content": f"Steve says: \"{steve_r}\""
        })

        # Log turn
        turn_entry = {
            "turn":    turn_n,
            "deepseek": ds_clean,
            "deepseek_latency": round(ds_lat, 2),
            "steve":   steve_r,
            "steve_latency_ms": round(s_lat * 1000, 1),
        }
        log["turns"].append(turn_entry)

        transcript_lines += [
            f"[Turn {turn_n}]",
            f"MARCUS ({ds_lat:.1f}s):",
            textwrap.fill(ds_clean, width=72, initial_indent="  ", subsequent_indent="  "),
            "",
            f"STEVE ({s_lat*1000:.0f}ms):",
            textwrap.fill(steve_r, width=72, initial_indent="  ", subsequent_indent="  "),
            "",
            "-" * 60,
            "",
        ]

    # ── Done — save logs ──────────────────────────────────────────────────────
    with open(log_json, 'w') as f:
        json.dump(log, f, indent=2)
    with open(log_txt, 'w') as f:
        f.write("\n".join(transcript_lines))

    print(f"\n[Sandbox] Done. Logs saved:")
    print(f"  {log_json}")
    print(f"  {log_txt}")

    # ── Analysis ─────────────────────────────────────────────────────────────
    analysis = analyse(log)
    print(f"\n{analysis}")

    # Append analysis to txt log
    with open(log_txt, 'a') as f:
        f.write("\n" + analysis + "\n")

    print(f"\n[Sandbox] Analysis appended to {log_txt}")
    print("[Sandbox] Sandbox killed.\n")


if __name__ == '__main__':
    main()
