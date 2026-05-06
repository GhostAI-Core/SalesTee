#!/usr/bin/env python3
"""
talk_tee.py — Voice conversation with Tee.

STT: faster-whisper (local, no API key)
TTS: edge-tts (Microsoft neural voices, free)

Usage:
    python talk_tee.py
    python talk_tee.py --voice en-GB-SoniaNeural   # change voice
    python talk_tee.py --debug                      # show retrieval details
    python talk_tee.py --list-voices                # show available voices

Press ENTER to speak. Tee responds, then immediately listens again.
Type 'quit' at the prompt to end the session.
Meta commands handled inline: "repeat that", "what time is it", "who are you".
"""

import os
import sys
import asyncio
import argparse
import tempfile
import textwrap
import threading
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

W = 72
SAMPLE_RATE = 16000
CHANNELS    = 1

# ── Voice config ──────────────────────────────────────────────────────────────

DEFAULT_VOICE = 'en-US-AvaMultilingualNeural'

VOICE_MENU = [
    ('en-US-AvaMultilingualNeural',    'US female  · multilingual'),
    ('en-US-BrianMultilingualNeural',  'US male    · multilingual'),
    ('en-ZA-LeahNeural',               'SA female'),
    ('en-ZA-LukeNeural',               'SA male'),
    ('zu-ZA-ThandoNeural',             'Zulu female'),
    ('zu-ZA-ThembaNeural',             'Zulu male'),
]


# ── Imports from chat_tee (reuse all routing logic) ───────────────────────────

from chat_tee import (
    tee_respond,
    opening_sequence,
    get_products,
    detect_product_intent,
    tee_print,
    _classify,
    build_query,
    W,
)


# ── Meta-question handler ─────────────────────────────────────────────────────

_META_TIME = {
    'what time is it', 'what is the time', 'tell me the time',
    'what\'s the time', 'whats the time', 'time please', 'the time',
}
_META_REPEAT = {
    'repeat that', 'say that again', 'can you repeat that', 'what did you say',
    'pardon', 'come again', 'i didn\'t catch that', 'i didnt catch that',
    'repeat', 'again',
}
_META_WHO = {
    'who are you', 'what are you', 'what is your name', 'your name',
    'are you a bot', 'are you ai', 'are you human', 'are you real',
    'who am i talking to', 'introduce yourself',
}


def handle_meta(text: str, last_response: str) -> str | None:
    """
    Returns a response string if the input is a meta command, else None.
    Meta commands are handled inline without touching the retrieval engine.
    """
    import datetime
    clean = text.lower().strip().rstrip('?.!')
    if clean in _META_TIME:
        now = datetime.datetime.now().strftime('%I:%M %p').lstrip('0')
        return f"It's {now}."
    if clean in _META_REPEAT:
        return last_response if last_response else "I haven't said anything yet."
    if clean in _META_WHO:
        return None  # let identity retrieval handle these — Tee has good cells for this
    return None


# ── STT ───────────────────────────────────────────────────────────────────────

_whisper_model = None

def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("[STT] Loading Whisper tiny.en (first run downloads ~75 MB)...", flush=True)
        _whisper_model = WhisperModel('tiny.en', device='cpu', compute_type='int8')
        print("[STT] Whisper ready.", flush=True)
    return _whisper_model


def record_until_enter() -> np.ndarray:
    """Record mic audio until user presses ENTER. Returns float32 array."""
    frames = []
    stop_event = threading.Event()

    def _stream_cb(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        callback=_stream_cb,
    )

    print(f"\n  {'─'*20}")
    print("  [Recording — press ENTER to stop]", end='', flush=True)
    with stream:
        input()   # blocks until ENTER

    audio = np.concatenate(frames, axis=0).flatten() if frames else np.array([], dtype='float32')
    return audio


def transcribe(audio: np.ndarray) -> str:
    """Transcribe float32 audio array → text string."""
    if len(audio) < SAMPLE_RATE * 0.3:   # ignore clips under 0.3 s
        return ''
    model = load_whisper()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, audio, SAMPLE_RATE)
        segments, _ = model.transcribe(tmp_path, language='en', beam_size=1)
        text = ' '.join(s.text for s in segments).strip()
        return text
    finally:
        os.unlink(tmp_path)


# ── TTS ───────────────────────────────────────────────────────────────────────

def speak(text: str, voice: str):
    """Convert text to speech and play it blocking."""
    asyncio.run(_speak_async(text, voice))


async def _speak_async(text: str, voice: str):
    import edge_tts
    import re

    # 1. Split into sentences to allow streaming start
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return

    # 2. Queue for pre-fetched audio files
    audio_queue = asyncio.Queue(maxsize=3)

    async def producer():
        """Generates audio files in the background."""
        for s in sentences:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                tmp_path = f.name
            try:
                communicate = edge_tts.Communicate(s, voice)
                await communicate.save(tmp_path)
                await audio_queue.put(tmp_path)
            except Exception as e:
                print(f"  [TTS Error] {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        await audio_queue.put(None)  # Sentinel

    async def consumer():
        """Plays audio files as they become available."""
        while True:
            path = await audio_queue.get()
            if path is None:
                break
            try:
                # Play in a thread to keep the event loop responsive
                await asyncio.to_thread(_play_audio, path)
            finally:
                try:
                    os.unlink(path)
                except Exception:
                    pass
            audio_queue.task_done()

    # Run producer and consumer in parallel
    await asyncio.gather(producer(), consumer())


def _play_audio(path: str):
    data, sr = sf.read(path, dtype='float32')
    sd.play(data, sr)
    sd.wait()


# ── Voice listing ─────────────────────────────────────────────────────────────

def list_voices():
    async def _list():
        import edge_tts
        voices = await edge_tts.list_voices()
        en = [v for v in voices if v['Locale'].startswith('en-')]
        print(f"\n{'─'*W}")
        print(f"  English edge-tts voices ({len(en)} total):\n")
        for v in sorted(en, key=lambda x: x['ShortName']):
            print(f"    {v['ShortName']:<35}  {v['Gender']}")
        print(f"{'─'*W}\n")
    asyncio.run(_list())


# ── Voice picker ─────────────────────────────────────────────────────────────

def pick_voice(preselected: str | None = None) -> str:
    """Show numbered voice menu at startup. Returns chosen voice name."""
    if preselected and preselected != DEFAULT_VOICE:
        return preselected   # explicit --voice flag passed, skip menu

    print(f"\n{'─'*W}")
    print("  Select Tee's voice:\n")
    print(f"    0  {VOICE_MENU[0][0]:<38} {VOICE_MENU[0][1]}  (default)")
    for i, (name, label) in enumerate(VOICE_MENU[1:], 1):
        print(f"    {i}  {name:<38} {label}")
    print()

    while True:
        raw = input("  Choice [0]: ").strip()
        if raw == '' or raw == '0':
            chosen = VOICE_MENU[0][0]
            break
        if raw.isdigit() and 1 <= int(raw) < len(VOICE_MENU):
            chosen = VOICE_MENU[int(raw)][0]
            break
        print(f"  Enter 0–{len(VOICE_MENU)-1}")

    print(f"\n  Voice set to: {chosen}")
    print(f"{'─'*W}\n")
    return chosen


# ── Display ───────────────────────────────────────────────────────────────────

def tee_say(text: str, voice: str):
    """Print and speak Tee's response."""
    wrapped = textwrap.fill(text, width=W - 6, subsequent_indent='       ')
    print(f"  \033[1mTee:\033[0m {wrapped}\n")
    speak(text, voice)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice',       default=None,
                        help='edge-tts voice name — skips the menu if set')
    parser.add_argument('--debug',       action='store_true',
                        help='Show retrieval details per turn')
    parser.add_argument('--list-voices', action='store_true',
                        help='List available English voices and exit')
    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    print(f"\n{'='*W}")
    print(f"  TEE — VOICE CHAT")
    print(f"  Hold ENTER → speak → release ENTER to send.")
    print(f"  Type 'quit' to end.")
    print(f"{'='*W}")

    voice = pick_voice(args.voice)

    print("[Loading Tee...]", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("ERROR: Tee's substrate failed to load.")
        sys.exit(1)

    # Pre-load Whisper
    load_whisper()

    cells       = bridge.substrate.methodology_cells
    n_identity  = sum(1 for c in cells.values() if 'identity'  in getattr(c, 'source_table', ''))
    n_reasoning = sum(1 for c in cells.values() if 'reasoning' in getattr(c, 'source_table', ''))
    n_product   = sum(1 for c in cells.values() if 'product'   in getattr(c, 'source_table', ''))
    print(f"[Tee online — {len(cells)} cells: {n_identity} identity / {n_reasoning} reasoning / {n_product} product]\n")
    print('─' * W)
    print()

    history:      list  = []
    used_cells:   deque = deque(maxlen=6)

    from learn import SessionLogger, log_miss, SIM_MISS
    session_log = SessionLogger(product=None)

    # Opening — reuse chat_tee logic, but speak the output
    import chat_tee as _ct
    _orig_print = _ct.tee_print

    def _speaking_print(text):
        _orig_print(text)
        speak(text, voice)

    _ct.tee_print = _speaking_print
    product_scope = opening_sequence(bridge, debug=args.debug)
    _ct.tee_print = _orig_print
    session_log.product = product_scope

    last_response = ''
    SILENCE_TIMEOUT = 10.0  # seconds before Tee checks in

    while True:
        print(f"\n  \033[90mPress ENTER to speak, or type:\033[0m", end=' ', flush=True)

        # Wait for input with a timeout — if user takes >2s, Tee checks in
        import select, msvcrt, time as _time

        t_start = _time.monotonic()
        got_input = False

        # On Windows we poll msvcrt for a keypress rather than select()
        while True:
            if msvcrt.kbhit():
                got_input = True
                break
            if _time.monotonic() - t_start > SILENCE_TIMEOUT:
                break
            _time.sleep(0.05)

        if not got_input:
            checkin = "Still there?"
            print(f"\n  \033[90m[Tee: {checkin}]\033[0m")
            speak(checkin, voice)
            # Wait a further 2 seconds — if still nothing, end the call
            t2 = _time.monotonic()
            got_input2 = False
            print(f"  \033[90mPress ENTER to speak, or type:\033[0m", end=' ', flush=True)
            while True:
                if msvcrt.kbhit():
                    got_input2 = True
                    break
                if _time.monotonic() - t2 > SILENCE_TIMEOUT:
                    break
                _time.sleep(0.05)
            if not got_input2:
                goodbye = "Okay, I'll leave it there. Call back anytime."
                print(f"\n  \033[90m[Tee: {goodbye}]\033[0m")
                speak(goodbye, voice)
                break
            try:
                raw_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                break
        else:
            try:
                raw_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                break

        if raw_input.lower() in ('quit', 'exit', 'q', 'bye'):
            break

        if raw_input == '':
            audio = record_until_enter()
            raw_input = transcribe(audio)
            if not raw_input:
                print("  [Nothing heard — try again]")
                continue
            print(f"  \033[90m[You said: {raw_input!r}]\033[0m")

        if not raw_input:
            continue

        # Meta commands — handled without retrieval
        meta = handle_meta(raw_input, last_response)
        if meta is not None:
            last_response = meta
            tee_say(meta, voice)
            continue

        if product_scope is None:
            products = get_products(bridge)
            if products:
                intent = detect_product_intent(raw_input, products)
                if intent:
                    product_scope = intent
                    session_log.product = product_scope
                    if args.debug:
                        print(f"  [auto-scoped to: {product_scope}]")

        response, cid, sim = tee_respond(bridge, raw_input, history, used_cells,
                                         product_scope, debug=args.debug)
        if cid:
            used_cells.append(cid)

        if sim < SIM_MISS:
            log_miss(raw_input, sim, product_scope)
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
