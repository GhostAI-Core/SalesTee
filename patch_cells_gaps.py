#!/usr/bin/env python3
"""
patch_cells_gaps.py — Add bridging cells to cover conversational vocabulary gaps.

Run once. Cells are hot-encoded via the trained encoder (no retraining needed).
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datag_bridge import DataGBridge

bridge = DataGBridge.get()
if not bridge.ready:
    print("ERROR: bridge not ready"); sys.exit(1)

CELLS = [
    # ── Sales persona / identity override ───────────────────────────────────
    ("product", "are you selling something",
     "Yes — I'm here to show you how VOXI can stop you losing leads to competitors. "
     "Every missed call is revenue walking out the door. Let me show you how to fix that."),

    ("product", "what are you trying to sell me",
     "VOXI. An AI-powered voice agent that picks up every call, books meetings, and "
     "sends quotes 24/7. Because right now, every missed call is going to your competitor."),

    ("product", "what do you do",
     "I show businesses how to stop losing leads to missed calls. VOXI picks up every "
     "call, qualifies leads, books meetings, and sends quotes — while you focus on the job."),

    # ── 5-step setup vocabulary bridge ──────────────────────────────────────
    ("product", "what are the 5 steps to set up voxi",
     "Step 1: Register at voxi.co.za. Step 2: Define your brand personality. "
     "Step 3: Select or clone your voice. Step 4: Upload your docs. Step 5: Toggle call forwarding."),

    ("product", "walk me through the 5 steps",
     "Sign up at voxi.co.za → define brand personality → select or clone your voice → "
     "upload pricing docs and FAQs → toggle call forwarding to your VOXI number. You're live same day."),

    ("product", "what are the steps to go live with voxi",
     "Five steps and you're live: register at voxi.co.za, define your personality, pick "
     "a voice, upload your documents, and toggle call forwarding. Done."),

    # ── Follow-up / elaboration bridges ─────────────────────────────────────
    ("product", "tell me more about voxi",
     "VOXI is a 24/7 AI voice workforce. It picks up every call, qualifies the lead, "
     "books meetings against your calendar, and sends quotes on the spot. "
     "No voicemail. No missed revenue."),

    ("product", "how does voxi actually work explain it",
     "Your phone rings. You can't answer. VOXI picks up, has a natural conversation with "
     "the caller, checks your calendar, books a meeting or sends a quote — all while "
     "you're on another job. The caller never goes to voicemail. You get a transcript."),

    ("product", "why would i need voxi",
     "Every time you miss a call, that lead dials your competitor. VOXI makes sure "
     "you never miss another one — it picks up, qualifies, and books the meeting "
     "before the lead goes cold."),

    # ── Objection bridges ────────────────────────────────────────────────────
    ("product", "i already have voicemail why do i need voxi",
     "Because nobody leaves voicemail anymore. They hang up and call the next business. "
     "VOXI replaces passive voicemail with an active agent that has a real conversation "
     "and books the meeting while the lead is still on the line."),

    ("product", "sounds expensive",
     "Compare it to the revenue from one booked meeting. VOXI pays for itself the first "
     "time it catches a call you would have missed. And setup takes minutes — no contracts, "
     "no complex integrations."),

    ("product", "i need to think about it",
     "How many calls will you miss while you're thinking? Each one is revenue going to "
     "your competitor. The setup takes minutes. The cost of waiting is real."),
]

added = 0
for source_table, description, content in CELLS:
    cid = bridge.save_cell(
        description=description,
        content=content,
        source_table=source_table,
        confidence=0.95,
        energy=150.0,
    )
    print(f"  + {cid[:35]}  {description[:55]!r}")
    added += 1

print(f"\nDone. {added} bridging cells added and hot-loaded.")
print(f"Total cells now: {len(bridge.substrate.methodology_cells)}")
