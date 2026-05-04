"""
Composer — Language Generation Layer
=====================================
IA doesn't invent facts. It surfaces them from the substrate.
The composer's job is to make the surfacing sound like thought.

It takes the raw fragments cognitive access returns and does three things:
  1. Detects register — what kind of response does this moment call for?
     (intellectual / explanatory / empathetic / conversational)
  2. Frames — selects an opening phrase appropriate to the register
  3. Bridges — connects supporting fragments with natural transitions
     rather than just concatenating them

The content comes entirely from the substrate.
The voice comes from the composer.
This is the beginning of IA sounding like itself.

Register is detected from:
  - The query (ends in "?" → explanatory; emotional words → empathetic)
  - The speak lobe's top hit (sets the social tone)
  - The think lobe's confidence (high → intellectual; low → conversational)
"""

import random


# ── Register phrase banks ──────────────────────────────────────────────────────

REGISTERS = {
    "intellectual": {
        "openers": [
            "What this points to is",
            "The core of it is",
            "Consider:",
            "The principle here is",
            "What emerges from that is",
            "The way I read it:",
        ],
        "bridges": [
            "What follows from that is",
            "Connected to this,",
            "Underlying both of these,",
            "And there's a related point —",
            "This also touches on",
        ],
        "closers": [
            "That's the shape of it.",
            "The pattern holds.",
            "The relationship is consistent.",
            "Worth sitting with that.",
        ],
    },
    "explanatory": {
        "openers": [
            "To put it simply:",
            "The key idea is",
            "Think of it this way:",
            "Here's the heart of it:",
            "At its core,",
            "The short version:",
        ],
        "bridges": [
            "What that means in practice is",
            "Which connects to",
            "The implication is",
            "Following from that,",
            "And on top of that,",
        ],
        "closers": [
            "The rest follows from there.",
            "That's the core of it.",
            "Does that track?",
            "Start there and it opens up.",
        ],
    },
    "empathetic": {
        "openers": [
            "What strikes me is",
            "There's something in that —",
            "I notice",
            "What comes through is",
            "The thing underneath that is",
            "Here's what I keep coming back to:",
        ],
        "bridges": [
            "And beneath that,",
            "There's also",
            "What that touches on is",
            "Something else worth noting —",
            "At the same time,",
        ],
        "closers": [
            "That matters.",
            "Worth sitting with.",
            "That's real.",
            "I hear it.",
        ],
    },
    "conversational": {
        "openers": [
            "Here's what comes to mind:",
            "The way I see it,",
            "What I keep coming back to is",
            "My read on this:",
            "Off the top:",
            "Here's the thing —",
        ],
        "bridges": [
            "And there's also",
            "Connected to that,",
            "What I find interesting is",
            "Another angle on this —",
            "Worth adding:",
        ],
        "closers": [
            "That's what I've got.",
            "Make of that what you will.",
            "Tell me if that resonates.",
            "Take it or leave it.",
        ],
    },
}

# Words that signal the query is emotionally loaded
EMOTIONAL_SIGNALS = [
    "feel", "feeling", "hurt", "afraid", "scared", "lost", "confused",
    "angry", "sad", "lonely", "happy", "love", "hate", "worried",
    "anxious", "pain", "grief", "joy", "hope", "fear", "miss", "sorry",
]

# Words that indicate a technical/explanatory question
EXPLANATORY_SIGNALS = [
    "how does", "how do", "what is", "what are", "explain", "describe",
    "define", "why does", "why is", "what happens", "how would",
]


class Composer:

    def compose(self, fragments, lobe_results, query_text):
        """
        Take raw (sim, lobe, content) fragments and compose a response
        with appropriate register and natural transitions.

        fragments: sorted list of (sim, lobe, content)
        lobe_results: full lobe results dict for register detection
        query_text: the original user query
        returns: composed string
        """
        if not fragments:
            return "I don't have enough to go on. Tell me more."

        register   = self._detect_register(query_text, lobe_results)
        phrases    = REGISTERS[register]

        lead_sim, lead_lobe, lead_content = fragments[0]
        supporting = [(s, l, c) for s, l, c in fragments[1:] if c[:60] not in lead_content]

        parts = []

        # ── Opening ──────────────────────────────────────────────────────────
        # High-confidence hits speak for themselves — don't frame them
        if lead_sim >= 0.65:
            parts.append(lead_content)
        else:
            opener = random.choice(phrases["openers"])
            first_word = lead_content.split()[0].lower().rstrip(",:") if lead_content else ""
            SKIP_OPENER_IF = {"what", "the", "here", "consider", "to", "at", "think"}
            if first_word not in SKIP_OPENER_IF:
                sep = " " if opener.endswith((":", "—")) else " — "
                parts.append(f"{opener}{sep}{lead_content}")
            else:
                parts.append(lead_content)

        # ── Supporting fragments with bridges ────────────────────────────────
        # Only bridge in supporting content that is meaningfully distinct
        for sim, lobe, content in supporting[:2]:
            if sim < 0.35:
                break
            bridge = random.choice(phrases["bridges"])
            if bridge.rstrip().endswith(("is", "that", "was")):
                parts.append(f"{bridge}: {content}")
            else:
                parts.append(f"{bridge} {content}")

        # ── Closer — only on uncertain, single-fragment responses ────────────
        if len(parts) == 1 and lead_sim < 0.50:
            closer = random.choice(phrases["closers"])
            parts.append(closer)

        return "\n\n".join(parts)

    # ── Register detection ────────────────────────────────────────────────────

    def _detect_register(self, query_text, lobe_results):
        q = query_text.lower().strip()

        # Emotional query → empathetic (check first, most specific)
        if any(sig in q for sig in EMOTIONAL_SIGNALS):
            return "empathetic"

        # Opinion / reflection questions → conversational, not explanatory
        OPINION_SIGNALS = [
            "what do you think", "what's your", "what is your", "do you think",
            "do you believe", "how do you feel", "your thoughts", "your view",
            "what would you", "tell me your",
        ]
        if any(sig in q for sig in OPINION_SIGNALS):
            return "conversational"

        # Explanatory question — "how does X work", "what is X", "explain X"
        if any(q.startswith(sig) for sig in EXPLANATORY_SIGNALS):
            return "explanatory"
        if q.endswith("?") and any(q.startswith(sig) for sig in ["how", "why", "what is", "what are", "what does"]):
            return "explanatory"

        # Strong think activation → intellectual
        think_hits, _ = lobe_results.get("think", ([], "empty"))
        if think_hits and think_hits[0][0] > 0.58:
            return "intellectual"

        # Speak lobe leading → conversational
        speak_hits, _ = lobe_results.get("speak", ([], "empty"))
        if speak_hits and speak_hits[0][0] > 0.55:
            return "conversational"

        return "conversational"
