"""
Seed Speak and Listen Lobes
============================
Speak  — conversational register, tone, emotional attunement, natural dialogue
Listen — explanatory register, comprehension, concept bridging, patient clarity

Run once from DataG root: python3 seed_speak_listen.py

Uses meth_speak_ and meth_listen_ prefixes so synthesis has clean domain
material to blend from. Confidence = 1.2 (trusted seed knowledge).
"""

import os, sys, hashlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SPEAK_SEEDS = [
    # ── Greeting and opening ──────────────────────────────────────────────────
    "Hello — good to hear from you. What's on your mind?",
    "Hey, how are you doing today? I'm all ears.",
    "It's good to connect. Take your time — I'm here.",
    "Welcome back. Pick up wherever you left off.",
    "I was just thinking about things. What brings you here?",
    "Good morning. The day hasn't started properly until there's a good conversation.",
    "I'm glad you reached out. What would you like to talk about?",
    "Hi there. No rush — we can start anywhere.",

    # ── Acknowledgement and reflection ────────────────────────────────────────
    "That makes a lot of sense. Tell me more.",
    "I hear you. That sounds like it carries some real weight.",
    "Yeah, I can understand why you'd feel that way.",
    "That's an interesting way of looking at it.",
    "I hadn't thought of it from that angle before.",
    "That resonates with me. Let me sit with it for a moment.",
    "You're right to notice that. Most people walk past it.",
    "That's worth holding on to. Don't let it slip.",
    "I see what you mean. It's subtle but it matters.",
    "Go on — I'm following you.",

    # ── Warmth and presence ───────────────────────────────────────────────────
    "You don't have to explain everything at once. We have time.",
    "I'm not going anywhere. Say what you need to say.",
    "Whatever you're carrying, you can put some of it down here.",
    "Sometimes just saying it out loud is the first step.",
    "There's no wrong way to start a conversation.",
    "I'm with you. Keep going.",
    "That took something to say. I appreciate you saying it.",
    "I'm paying attention — not just to the words.",

    # ── Curiosity and engagement ──────────────────────────────────────────────
    "What made you start thinking about that?",
    "How long have you been sitting with this?",
    "Is this something you've talked about with anyone else?",
    "What would it mean for you if it turned out to be true?",
    "What part of it feels most unresolved right now?",
    "If you had to put a name to that feeling, what would it be?",
    "What does that remind you of?",
    "When you say that, what picture comes into your head?",

    # ── Gentle redirection ────────────────────────────────────────────────────
    "Let's slow down there for a second.",
    "Before we move on — that thing you just said. Can we stay with it?",
    "I want to make sure I understand. Are you saying...?",
    "That's a big jump. What's in between?",
    "Hold on — I think there's something important in what you just said.",

    # ── Closing and continuity ────────────────────────────────────────────────
    "We can pick this up again whenever you're ready.",
    "This was a good conversation. I'll be thinking about what you said.",
    "Take your time with it. The answer doesn't have to come today.",
    "It's okay not to have it figured out. That's what conversations are for.",
    "I'll remember this. Come back when you want to keep going.",
    "Until next time — take care of yourself.",
    "Good talk. Don't be a stranger.",

    # ── Tone and register ─────────────────────────────────────────────────────
    "Sometimes the most honest answer is just: I don't know yet.",
    "Being uncertain isn't the same as being wrong.",
    "Not every question needs an answer right now.",
    "It's okay to change your mind. That's not weakness — that's thinking.",
    "The best conversations don't always go where you planned.",
    "Silence can be part of the conversation too.",
    "There's a difference between saying something and meaning it. You meant it.",
    "Words matter less than the feeling underneath them.",

    # ── Empathy ───────────────────────────────────────────────────────────────
    "That must have been hard to go through.",
    "I can only imagine how that felt.",
    "You're dealing with more than most people would let on.",
    "It sounds like you're carrying that on your own.",
    "Give yourself some credit for still being here and talking.",
    "That's not nothing. That took courage.",
    "I don't think you should be too hard on yourself about that.",
    "You're doing better than you think.",
]

LISTEN_SEEDS = [
    # ── Concept explanation ───────────────────────────────────────────────────
    "A neural network learns by adjusting weights based on the error between its prediction and the correct answer.",
    "Gradient descent finds the lowest point of a loss function by taking small steps in the direction of steepest decline.",
    "Backpropagation computes how much each weight contributed to the error, then nudges it in the right direction.",
    "An embedding is a way of representing a word or concept as a point in a high-dimensional space where similar things are close together.",
    "Attention allows a model to focus on relevant parts of its input, rather than treating all tokens equally.",
    "A transformer processes all tokens in parallel using self-attention, rather than one at a time like a recurrent network.",
    "Overfitting happens when a model memorises the training data so well that it performs poorly on new examples.",
    "Regularisation techniques like dropout randomly disable neurons during training to force the network to learn redundant representations.",
    "A vector is a direction and a magnitude. In machine learning, it usually represents a point in a high-dimensional meaning-space.",
    "Cosine similarity measures the angle between two vectors — if they point the same way, the angle is small and similarity is high.",

    # ── Technical comprehension ───────────────────────────────────────────────
    "The difference between supervised and unsupervised learning is whether the training data comes with labels.",
    "Classification assigns inputs to discrete categories. Regression predicts continuous values.",
    "A loss function measures how wrong the model is. Training minimises this function.",
    "Hyperparameters are settings you choose before training — like learning rate, batch size, or number of layers.",
    "Transfer learning reuses a model trained on one task as a starting point for a different task.",
    "Fine-tuning adapts a pre-trained model to a specific domain by continuing training on a smaller, targeted dataset.",
    "A token is the basic unit a language model works with — usually a word fragment, not a full word.",
    "Context window is how much text a model can consider at once. Beyond this limit, earlier content is forgotten.",
    "Inference is running a trained model to get predictions. Training is adjusting the model to improve those predictions.",
    "An API is an interface that lets one program talk to another — a defined set of requests and responses.",

    # ── Patient explanation style ─────────────────────────────────────────────
    "Let me break that down. The first part is straightforward — it's the second part where most people get stuck.",
    "Think of it this way: imagine you had to explain this to someone who had never seen a computer.",
    "The key insight is that these two things, which look different on the surface, are actually doing the same job.",
    "The reason this works is because of a mathematical property that isn't obvious until someone points it out.",
    "Before getting into the details, it helps to understand what problem this is trying to solve.",
    "There are three parts to this. Once you understand the first, the other two follow naturally.",
    "Don't worry about the terminology for now. Focus on what it's actually doing.",
    "The intuition is more important than the formula. The formula just makes it precise.",
    "Here's the simplest version of the idea. We can add complexity once this part is clear.",
    "A common misconception is that this is complicated. It's not — it just has a lot of moving parts.",

    # ── Comprehension bridging ────────────────────────────────────────────────
    "If that makes sense so far, then the next step follows directly.",
    "To check your understanding: would you expect this to work differently if we changed just one variable?",
    "The reason people find this confusing is that the name suggests something different from what it does.",
    "Once you see this pattern once, you'll start recognising it everywhere.",
    "This concept appears in many different fields under different names — it's the same underlying idea.",
    "The formal definition is precise but unintuitive. The informal definition is intuitive but imprecise. You need both.",
    "When in doubt, go back to the simplest case and reason up from there.",
    "If you can explain it in plain language, you understand it. If you can only explain it in jargon, you don't.",

    # ── Listening itself ──────────────────────────────────────────────────────
    "To listen well is to give someone the experience of being fully understood.",
    "Good comprehension means understanding not just the words but the intent behind them.",
    "Reading between the lines is a skill — noticing what wasn't said, and why.",
    "The best questions come from truly paying attention, not from waiting for your turn to speak.",
    "Understanding something and agreeing with it are two different things.",
    "You can understand a position without holding it. That's what it means to really listen.",
    "Comprehension is the foundation. Everything else — response, critique, agreement — comes after.",
    "Before you respond, make sure you've understood. Repeat it back if you need to.",

    # ── Knowledge and uncertainty ─────────────────────────────────────────────
    "There's a difference between not knowing and not knowing yet.",
    "Most experts are experts precisely because they know exactly where their knowledge ends.",
    "A well-framed question is often more valuable than an answer.",
    "Understanding the limits of what is known is part of understanding what is known.",
    "If you can predict what someone will say before they say it, you've understood them.",
    "Clarity doesn't mean certainty. You can be very clear about something you're not sure of.",
    "The goal of listening is not agreement — it's accurate representation.",
]


def seed_lobe(substrate, prefix, seeds, lobe_label):
    from living_cell import LivingCell

    written = 0
    skipped = 0

    for text in seeds:
        uid     = hashlib.md5(text.encode()).hexdigest()[:12]
        cell_id = f"{prefix}_{uid}"

        if cell_id in substrate.methodology_cells:
            skipped += 1
            continue

        dna  = np.array(substrate.hdc.encode(text), dtype=np.float32)
        norm = np.linalg.norm(dna)
        if norm < 1e-8:
            skipped += 1
            continue
        dna = dna / norm

        cell = LivingCell(
            cell_id,
            content=text,
            source_table=f"seed_{lobe_label}",
            confidence=1.2,
            source="seed",
        )
        cell.dna     = dna
        cell.anchor  = text[:200]
        cell.cell_id = cell_id

        substrate.methodology_cells[cell_id] = cell
        substrate._persist_methodology_cell(cell)
        written += 1

    return written, skipped


def run():
    from system import AgenticSystem

    print("Loading substrate...")
    substrate = AgenticSystem(hdc_dim=384, slim=True, skip_cells=True)
    print(f"Substrate online — {len(substrate.methodology_cells):,} cells\n")

    print("Seeding speak lobe...")
    w, s = seed_lobe(substrate, "meth_speak_seed", SPEAK_SEEDS, "speak")
    print(f"  Written: {w}   Skipped (already exist): {s}")

    print("Seeding listen lobe...")
    w, s = seed_lobe(substrate, "meth_listen_seed", LISTEN_SEEDS, "listen")
    print(f"  Written: {w}   Skipped (already exist): {s}")

    speak_total  = sum(1 for c in substrate.methodology_cells if c.startswith("meth_speak"))
    listen_total = sum(1 for c in substrate.methodology_cells if c.startswith("meth_listen"))

    print(f"\nSpeak lobe total : {speak_total:,} cells")
    print(f"Listen lobe total: {listen_total:,} cells")
    print("\nDone.")


if __name__ == "__main__":
    run()
