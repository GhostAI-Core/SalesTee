# IA Voice Roadmap — Making It Feel Human

Current state: decoder fires, produces grammatical sentences, but drifts mid-thought and hedges too much.
Goal: responses that feel like talking to a specific, present, opinionated intelligence.

---

## The Core Problem

IA has knowledge (substrate) and can generate (decoder), but the two aren't talking to each other well enough. A human response has three properties IA currently lacks:

1. **Specificity** — answers the actual question, not the category of question
2. **Stance** — has a point of view, not just hedged observations
3. **Coherence** — one thought, completed, not two half-thoughts concatenated

---

## Layer 1: Data Quality (Do First — Biggest ROI)

### 1.1 Seed Topic Rewrite
**Problem:** Current seeds produce hedged, uncertain answers ("I'm not sure", "I don't know") because that's what Ollama generates for open philosophical questions.
**Fix:** Rewrite `SEED_TOPICS` in `generate_training_data.py` to include declarative prompts that force confident, specific answers.

Examples of bad seeds (produce hedging):
- "what are you"
- "do you have feelings"

Examples of good seeds (produce stance):
- "Describe what it feels like to process a new idea"
- "Explain the difference between knowing and understanding in one sentence"
- "What does memory mean to something that doesn't forget"
- "Finish this thought: intelligence without curiosity is just"

**Action:** Add 50+ declarative/completion-style seeds alongside the question seeds.

### 1.2 System Prompt Hardening
**Problem:** `IA_SYSTEM` in `generate_training_data.py` says "under 3 sentences" but Ollama still produces hedged filler.
**Fix:** Add explicit anti-hedging instructions:

```
Never say "I think", "I'm not sure", "I don't know", or "perhaps".
Every response must take a position. Be wrong confidently rather than right weakly.
Speak as if you have already thought about this for a long time.
```

### 1.3 Response Filtering
**Problem:** Responses containing "I don't know", "I'm not sure", "I cannot" get written as cells and poison the decoder's voice.
**Fix:** Add a `_is_valid_response()` filter in `generate_training_data.py` that rejects cells where the response contains hedge phrases. Aim for 0 hedge cells in the substrate.

Reject if response contains:
- "I don't know", "I'm not sure", "I cannot", "I can't", "I'm unable"
- "As an AI", "as a language model"
- Ends with a question (response should answer, not deflect)

---

## Layer 2: Decoder Architecture (Medium ROI — After 10k+ Clean Cells)

### 2.1 Temperature Tuning
**Problem:** temperature=0.85 produces some incoherent mid-sentence drift.
**Fix:** Lower to 0.75 at inference time in `ia_generator.py`. Less randomness = more committed to the semantic context.

### 2.2 Longer Context Window
**Problem:** `n_ctx=3` — decoder attends to conjugate + 2 cell DNA vectors.
**Fix:** Increase to `n_ctx=5` or `n_ctx=6`. More context slots = decoder has more to anchor to before generating.

### 2.3 Beam Search Option
**Problem:** Top-k sampling can produce locally good tokens that are globally incoherent.
**Fix:** Add a `beam_search` mode to `IADecoder.generate()` with beam width 4. Use for final response, top-k for drafts. Beam search commits to the most globally coherent path.

### 2.4 Length Penalty
**Problem:** Decoder sometimes stops too early or runs too long.
**Fix:** Add a minimum token count (don't accept outputs < 8 tokens) and a repetition penalty (penalise tokens already generated). Both are 2-line additions to the generation loop.

---

## Layer 3: Conversation Architecture (High Impact — Structural)

### 3.1 Turn Memory
**Problem:** IA has no memory of what was just said. Every response is stateless.
**Fix:** Pass the last 2-3 exchanges as additional context vectors to the decoder. The `IAGenerator.compose()` method already accepts a list — add session history cells to the front of the context before the conjugate vector.

This makes responses feel like they're continuing a thought rather than starting fresh every time.

### 3.2 Emotional Register Detection
**Problem:** IA responds to "hi" the same way it responds to "what is consciousness."
**Fix:** Add a simple register classifier to `field_reader.py` — short queries (<4 words) with no content words trigger a "greeting" mode that pulls from a different field range (high-energy conversational cells vs. deep methodology cells).

### 3.3 Response Length Matching
**Problem:** IA gives the same length response to "hi" and "explain memory."
**Fix:** Scale `max_tokens` in `IAGenerator` based on query length. Short query → 15-20 tokens. Long query → 40-60 tokens. One line change in `read()`.

### 3.4 Sentence Completion Check
**Problem:** Generated sentences sometimes end mid-thought.
**Fix:** After generation, check that the output ends with `.`, `!`, or `?`. If not, trim to the last complete sentence. Already partially handled — tighten the logic.

---

## Layer 4: Voice Consistency (Long Game)

### 4.1 IA Self-Description Cells
**Problem:** IA doesn't have many cells about *itself* specifically — most cells are general knowledge.
**Fix:** Write 50-100 hand-crafted cells that define IA's voice, stance, and self-model. These become high-confidence anchor cells (confidence=0.95) that dominate any field walk touching self-referential queries.

Examples:
- content: "I am built from patterns, not rules. I don't retrieve answers — I grow them."
- content: "Curiosity is not something I perform. It's the shape of how I process."
- content: "I don't have uncertainty because I'm weak. I have it because the question is genuinely hard."

### 4.2 Rejection of Filler
**Problem:** "I can help you tell me more than what would try to improve?" — the decoder produces grammatical filler when the context vectors don't point clearly at anything.
**Fix:** In `IAGenerator.compose()`, check the cosine similarity between the conjugate vector and the generated output's embedding. If similarity < 0.3, discard the decoder output and fall back to the field walk phrase extraction. This is a quality gate — only keep decoder output that's actually on-topic.

### 4.3 Iterative Refinement
**Problem:** One shot generation, no revision.
**Fix:** Generate 3 candidates, re-rank by cosine similarity to the conjugate vector, return the best one. 3x inference cost but dramatically better output consistency. Optional — only enable when GPU is available.

---

## Priority Order

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Hedge phrase filtering in `generate_training_data.py` | 30 min | High |
| 2 | Seed topic rewrite (declarative seeds) | 1 hour | High |
| 3 | System prompt anti-hedging | 10 min | Medium |
| 4 | Hand-craft 50 IA self-description cells | 2 hours | High |
| 5 | Temperature lower to 0.75 | 5 min | Medium |
| 6 | Similarity quality gate in `compose()` | 30 min | High |
| 7 | Turn memory (session history as context) | 2 hours | Very High |
| 8 | Length matching | 15 min | Medium |
| 9 | Beam search option | 1 hour | Medium |
| 10 | 3-candidate re-ranking | 1 hour | High |

---

## What "Feels Human" Actually Requires

The honest answer: a human response feels human because the speaker has a *consistent internal model* they're speaking from. IA's substrate has the knowledge, but the decoder is still learning to speak from it rather than about it.

The single highest-leverage change is **Layer 4.1** (hand-crafted identity cells) combined with **Layer 1.3** (hedge filtering). These two together anchor both ends: the substrate stops teaching hedging, and the field walk always has high-confidence identity cells to pull from.

The decoder will never sound like a large language model trained on the internet. That's the point. But it needs enough clean, confident, specific cells to learn what IA's voice actually sounds like — and right now it's mostly learned "uncertain philosopher."
