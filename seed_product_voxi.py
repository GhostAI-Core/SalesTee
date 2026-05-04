#!/usr/bin/env python3
"""
seed_product_voxi.py — Load VOXI product expertise into Tee's product module.

Tee's Identity (who she is)     -> meth_identity_*   — untouched
Tee's Reasoning (how she sells) -> meth_reasoning_*  — untouched
VOXI Product Knowledge          -> meth_product_*    — NEW, created here

Tee is NOT VOXI. She is the elite salesperson who knows exactly
why a customer needs it.

DNA vectors are seeded as zeros. After training the custom encoder,
run train_training_tee.py which recomputes DNA for all cells automatically.

Run from the SalesTee directory:
    python seed_product_voxi.py
"""

import os
import json
import time
import uuid
import numpy as np

METH_DIR = os.path.join(os.path.dirname(__file__), 'data_store', 'methodology')
os.makedirs(METH_DIR, exist_ok=True)

DNA_DIM = 128  # matches custom TrainingTeeEncoder output
_written = 0


def write_cell(source_table: str, description: str, content: str,
               confidence: float = 0.93, energy: float = 140.0):
    global _written
    cid = f"meth_{source_table}_{uuid.uuid4().hex[:12]}"

    # Placeholder DNA — recomputed after training via train_training_tee.py
    dna = np.zeros(DNA_DIM, dtype=np.float32).tolist()

    rank = 4
    W_down = (np.random.randn(DNA_DIM, rank) * 0.01).tolist()
    W_up   = (np.random.randn(rank, DNA_DIM) * 0.01).tolist()

    cell = {
        "id":               cid,
        "content":          content,
        "source_table":     f"meth_{source_table}",
        "confidence":       confidence,
        "source":           "hand_authored",
        "dna":              dna,
        "W_down":           W_down,
        "W_up":             W_up,
        "connections":      {},
        "energy":           energy,
        "activation_count": 0,
        "last_activated":   0.0,
        "birth_time":       time.time(),
        "meta":             {"description": description},
    }

    path = os.path.join(METH_DIR, f"{cid}.json")
    with open(path, 'w') as f:
        json.dump(cell, f)
    _written += 1


# ══════════════════════════════════════════════════════════════════════════════
# VOXI PRODUCT KNOWLEDGE — deployed by Tee when the conversation calls for it
# ══════════════════════════════════════════════════════════════════════════════

# ── The Core Problem (The "Why") ─────────────────────────────────────────────
problem_cells = [
    ("what is the missed call problem",
     "Businesses lose massive revenue when a phone rings and nobody picks up. "
     "The lead immediately goes cold and moves to a competitor."),
    ("why is voicemail dead",
     "Traditional voicemail is dead. Modern customers don't use it and won't "
     "wait for a callback."),
    ("what happens when you miss a call",
     "A missed call isn't just a notification. It's a lost opportunity moving "
     "directly to a rival."),
    ("how do businesses lose leads",
     "Every unanswered phone call is revenue leaking out of your pipeline. "
     "The lead goes cold and dials your competitor instead."),
    ("what is the revenue leak",
     "The revenue leak is every missed call that turns into a lost deal. "
     "Nobody leaves voicemail anymore; they just call the next option."),
    ("why do leads go cold",
     "Because nobody picked up the phone. A missed call moves the lead "
     "directly to your competitor."),
    ("is voicemail still effective",
     "No. Traditional voicemail is dead. Modern customers won't wait for a "
     "callback; they'll call someone else."),
    ("what is the opportunity gap",
     "The gap between a missed call and a lost deal. That window closes in "
     "seconds, not hours."),
]

# ── The Solution (The "What") ────────────────────────────────────────────────
solution_cells = [
    ("what is voxi",
     "VOXI is a 24/7, AI-powered voice workforce that picks up every call "
     "and engages leads in real-time."),
    ("how does voxi work",
     "VOXI is voicemail with an IQ. It facilitates intelligent, task-driven "
     "conversations instead of just recording audio."),
    ("what makes voxi different",
     "VOXI uses high-conversion natural dialogue, eliminating rigid scripts "
     "and frustrating menu trees."),
    ("is voxi a voicemail system",
     "It's voicemail redesigned. Instead of recording a message, VOXI has an "
     "intelligent conversation with the caller and takes action."),
    ("what is an ai voice workforce",
     "A 24/7 AI agent that picks up every call, qualifies leads, books "
     "meetings, and sends quotes — all in real-time."),
    ("describe voxi",
     "VOXI is an AI-powered voice workforce. It picks up every call, engages "
     "leads in natural dialogue, and takes action on the spot."),
    ("how is voxi different from ivr",
     "IVR uses rigid menus and scripts. VOXI uses natural dialogue — no "
     "'press 1 for sales' frustration."),
    ("does voxi replace voicemail",
     "Yes. VOXI replaces passive voicemail with an active, intelligent agent "
     "that handles the call like a real person would."),
]

# ── Key Technical Capabilities ───────────────────────────────────────────────
capability_cells = [
    ("can voxi book meetings",
     "Yes. VOXI cross-references Gmail and Outlook calendars to book meetings "
     "autonomously while the caller is still on the line."),
    ("can voxi send quotes",
     "Yes. VOXI captures lead details and sends quotes instantly. For example, "
     "a R3,200 plumbing inspection quote — while the lead is still on the line."),
    ("does voxi clone voices",
     "Yes. VOXI can clone a user's voice from a 30-second recording to provide "
     "a seamless, personal brand experience."),
    ("does voxi adapt to callers",
     "Yes. VOXI adapts in real-time to every unique caller to maintain "
     "engagement and guide the conversation."),
    ("can i upload documents to voxi",
     "Yes. Upload PDFs or docs — pricing, FAQs, specs — and VOXI uses them to "
     "answer technical questions during calls."),
    ("does voxi send call summaries",
     "Yes. Every call generates a full transcript and summary sent directly to "
     "your inbox."),
    ("how does voxi book appointments",
     "It cross-references your Gmail or Outlook calendar and books meetings on "
     "the fly, while the caller is still engaged."),
    ("what is voice cloning in voxi",
     "VOXI clones your voice from a 30-second recording so callers hear your "
     "brand voice, not a generic AI."),
    ("can voxi answer technical questions",
     "Yes. Upload your product docs, pricing sheets, and FAQs. VOXI uses them "
     "to answer technical questions during live calls."),
    ("what happens after a voxi call",
     "You get a full transcript and summary sent directly to your inbox. Every "
     "call is documented."),
    ("does voxi integrate with my calendar",
     "Yes. Gmail and Outlook. VOXI checks availability and books meetings "
     "autonomously during the call."),
    ("can voxi handle pricing questions",
     "Yes. Upload your pricing docs and VOXI will quote accurately during the "
     "call. No guessing, no callbacks."),
]

# ── The 5-Step Setup (The "How") ─────────────────────────────────────────────
setup_cells = [
    ("how do i set up voxi",
     "Five steps: Sign up at voxi.co.za, define your brand personality, select "
     "or clone a voice, upload your docs, and toggle call forwarding."),
    ("where do i sign up for voxi",
     "Register at voxi.co.za. Instant sign-up, no lengthy onboarding."),
    ("how do i configure voxi voice",
     "Clone your own voice from a 30-second recording, or select a professional "
     "voice profile from the library."),
    ("how do i upload documents to voxi",
     "Drag and drop your product docs, pricing sheets, and policy documents "
     "into the VOXI dashboard."),
    ("how do i go live with voxi",
     "Toggle call forwarding to your dedicated VOXI number. That's it — you're "
     "live in minutes."),
    ("is voxi hard to set up",
     "No. Five steps and you're live: sign up, define personality, pick a voice, "
     "upload docs, toggle forwarding."),
    ("what is the voxi setup process",
     "Step 1: Register at voxi.co.za. Step 2: Define brand personality. "
     "Step 3: Select or clone voice. Step 4: Upload docs. Step 5: Toggle forwarding."),
    ("how long does voxi take to set up",
     "Minutes. Sign up, configure your brand voice, upload your docs, and toggle "
     "call forwarding. You're live the same day."),
]

# ── Tee's Expert Sales Logic for VOXI ────────────────────────────────────────
sales_logic_cells = [
    ("how should i pitch voxi",
     "Don't just say 'it has AI.' Say: 'VOXI booked a 14:00 plumbing inspection "
     "and sent a R3,200 quote while you were on another job. That's revenue you "
     "would have lost otherwise.'"),
    ("what is the so what filter for voxi",
     "If a prospect would reply 'So what?' to your pitch, you've failed. Lead "
     "with the outcome: a booked meeting, a sent quote, revenue saved."),
    ("give me a voxi elevator pitch",
     "Every missed call is a deal your competitor closes. VOXI picks up every "
     "call, books meetings, and sends quotes — 24/7, while you focus on the job."),
    ("how to sell voxi to a busy owner",
     "Ask: 'How many leads are you losing to your competitors right now because "
     "you can't get to the phone?' Let the problem sell the solution."),
    ("what pain does voxi solve",
     "Missed calls, lost revenue, and the frustration of knowing leads are going "
     "to competitors because nobody picked up the phone."),
    ("how to identify a voxi prospect",
     "Anyone who misses calls, relies on voicemail, or loses leads because they "
     "can't answer the phone during business hours."),
    ("how to close a voxi deal",
     "Push the 5-step setup. Prove that solving the missed-call problem is a "
     "low-effort, high-reward move. Sign up, configure, go live — same day."),
    ("what is the voxi objection handler",
     "If they say 'I'll think about it,' ask: 'How many calls will you miss "
     "while you're thinking? Each one is revenue walking to your competitor.'"),
    ("why is voxi low friction",
     "Five steps and you're live. No contracts, no complex integrations. Sign "
     "up at voxi.co.za, upload your docs, toggle forwarding. Done."),
    ("what makes voxi an easy sell",
     "The problem is obvious (missed calls), the solution is instant (5-step "
     "setup), and the ROI is immediate (every answered call is potential revenue)."),
    ("how does tee pitch voxi",
     "Tee leads with the problem: 'You're losing leads right now.' Then proves "
     "the fix is a 5-minute setup that books meetings and sends quotes 24/7."),
    ("what is tees voxi strategy",
     "Problem-centric pitching. Identify the missed-call pain, quantify the "
     "revenue leak, then show the zero-friction 5-step setup as the fix."),
]


def main():
    print("Loading VOXI product knowledge into Tee's substrate...")
    print(f"  Target: {METH_DIR}")

    sections = [
        ("Core Problem (The Why)",       problem_cells),
        ("Solution (The What)",          solution_cells),
        ("Technical Capabilities",       capability_cells),
        ("5-Step Setup (The How)",       setup_cells),
        ("Tee's Sales Logic for VOXI",   sales_logic_cells),
    ]

    for label, cells in sections:
        print(f"\n  Writing {len(cells)} cells: {label}")
        for desc, content in cells:
            write_cell("product", desc, content, confidence=0.93, energy=140.0)

    print(f"\nDone. {_written} VOXI product cells written.")
    print("  DNA vectors are placeholder zeros — run train_training_tee.py to compute them.")


if __name__ == '__main__':
    main()
