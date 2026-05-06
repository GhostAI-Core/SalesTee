#!/usr/bin/env python3
"""
talk_voxi.py — VOXI-specific voice agent.

Hard-scoped to the VOXI product. Opens with a spoken intro retrieved
from the substrate, then enters the standard VAD voice loop.

Usage:
    python talk_voxi.py
    python talk_voxi.py --voice en-ZA-LeahNeural
    python talk_voxi.py --debug
"""

import os
import sys
import textwrap
import argparse
from collections import deque

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from talk_tee import (
    speak, tee_say, load_whisper, record_vad, transcribe,
    pick_voice, handle_meta,
    DEFAULT_VOICE, W,
)
from chat_tee import (
    tee_respond, get_products, tee_print, _classify, build_query,
)

PRODUCT_SCOPE = 'voxi'


def voxi_intro(bridge, voice: str, debug: bool = False) -> str:
    """
    Speak the VOXI intro: a fixed greeting + the best matching intro cell.
    Returns the spoken text so it can seed last_response.
    """
    greeting = "Hi, I'm Tee. Let me tell you about VOXI."
    tee_say(greeting, voice)

    # Pull the best intro cell for VOXI
    intro_query = "what is voxi and how does it work"
    hits = bridge.top_cells(intro_query, k=10)
    filtered = [
        (s, cid, c) for s, cid, c in hits
        if getattr(c, 'source_table', '') == f'meth_product_{PRODUCT_SCOPE}'
    ]
    if filtered:
        best_sim, best_cid, best_cell = filtered[0]
        content = (getattr(best_cell, 'content', '') or '').strip()
        if content and best_sim >= 0.20:
            if debug:
                print(f"  \033[90m[intro sim={best_sim:.3f} cid={best_cid}]\033[0m")
            tee_say(content, voice)
            return greeting + ' ' + content

    return greeting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice', default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"\n{'='*W}")
    print(f"  TEE — VOXI VOICE AGENT")
    print(f"  Speak naturally — Tee listens automatically.")
    print(f"{'='*W}")

    voice = pick_voice(args.voice)

    print("[Loading Tee...]", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("ERROR: substrate failed to load.")
        sys.exit(1)

    load_whisper()

    cells = bridge.substrate.methodology_cells
    n_product = sum(1 for c in cells.values()
                    if getattr(c, 'source_table', '') == f'meth_product_{PRODUCT_SCOPE}')
    print(f"[Tee online — {len(cells)} cells total, {n_product} VOXI cells]\n")
    print('─' * W)
    print()

    history:    list  = []
    used_cells: deque = deque(maxlen=6)

    from learn import SessionLogger, log_miss, SIM_MISS
    session_log = SessionLogger(product=PRODUCT_SCOPE)

    last_response = voxi_intro(bridge, voice, debug=args.debug)

    LISTEN_TIMEOUT = 10.0

    while True:
        audio = record_vad(silence_sec=1.5, max_sec=LISTEN_TIMEOUT)

        if len(audio) == 0:
            checkin = "Still there?"
            tee_say(checkin, voice)
            audio2 = record_vad(silence_sec=1.5, max_sec=LISTEN_TIMEOUT)
            if len(audio2) == 0:
                goodbye = "Okay, I'll leave it there. Call back anytime."
                tee_say(goodbye, voice)
                break
            audio = audio2

        raw_input = transcribe(audio)
        if not raw_input:
            print("  \033[90m[Nothing heard — listening again]\033[0m")
            continue
        print(f"  \033[90m[You: {raw_input}]\033[0m")

        meta = handle_meta(raw_input, last_response)
        if meta is not None:
            last_response = meta
            tee_say(meta, voice)
            continue

        response, cid, sim = tee_respond(bridge, raw_input, history, used_cells,
                                         PRODUCT_SCOPE, debug=args.debug)
        if cid:
            used_cells.append(cid)

        if sim < SIM_MISS:
            log_miss(raw_input, sim, PRODUCT_SCOPE)
        else:
            session_log.log(user=raw_input, tee=response, sim=sim, cell_id=cid)

        history.append({'user': raw_input, 'tee': response})
        last_response = response
        tee_say(response, voice)

    session_log.close()
    print(f"\n{'─'*W}")
    print(f"  {len(history)} turns. Session ended.")
    print(f"{'='*W}\n")


if __name__ == '__main__':
    main()
