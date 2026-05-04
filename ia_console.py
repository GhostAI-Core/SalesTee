"""
IA Console — Cognitive Access Interface
========================================
Talk to IA. IA uses its DataG substrate through cognitive access —
not retrieval. It checks, fact-checks, corrects weights, and creates
knowledge where gaps exist.

Think lobe is unrestricted: it draws from all domains.
Output is routed through speak. Fidelity drops at the lobe boundary.

Run from DataG root: python3 ia_console.py

Commands:
  /census          — show current lobe populations
  /introspect      — show what activated and what status each lobe returned
  /lobe <name>     — access a specific lobe with the last query
  /cells <n>       — show top n activated cells in full
  /learn           — show exchanges logged this session
  /exit            — quit
"""

import os, sys, json, hashlib, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Lobe configuration ─────────────────────────────────────────────────────────
LOBE_PREFIXES = {
    "speak":  ["meth_conv_deep", "meth_conv_groq", "meth_conv", "meth_conversational",
               "meth_speech", "meth_speak", "meth_synthesis_speak"],
    "listen": ["meth_conv_tech", "meth_synthesis_listen", "meth_listen"],
    "mind":   ["meth_mind", "meth_claim", "meth_creator", "meth_english_linguistics",
               "meth_general_comprehension", "meth_document_parsing", "meth_synthesis_mind"],
    "think":  ["meth_"],   # unrestricted — all domains
    "code":   ["meth_code", "meth_synthesis_code"],
    "visual": ["meth_visual", "meth_synthesis_visual"],
}

PREFIX_TO_LOBE = [
    ("meth_synthesis_speak",        "speak"),
    ("meth_synthesis_listen",       "listen"),
    ("meth_synthesis_mind",         "mind"),
    ("meth_synthesis_think",        "think"),
    ("meth_synthesis_code",         "code"),
    ("meth_synthesis_visual",       "visual"),
    ("meth_synthesis_idle",         "think"),
    ("meth_synthesis",              "think"),
    ("meth_conv_tech",              "listen"),
    ("meth_conv_deep",              "speak"),
    ("meth_conv_groq",              "speak"),
    ("meth_conv",                   "speak"),
    ("meth_conversational",         "speak"),
    ("meth_speech",                 "speak"),
    ("meth_speak",                  "speak"),
    ("meth_listen",                 "listen"),
    ("meth_mind",                   "mind"),
    ("meth_claim",                  "mind"),
    ("meth_creator",                "mind"),
    ("meth_english_linguistics",    "mind"),
    ("meth_general_comprehension",  "mind"),
    ("meth_document_parsing",       "mind"),
    ("meth_code",                   "code"),
    ("meth_think",                  "think"),
    ("meth_architect",              "think"),
    ("meth_mathematical",           "think"),
    ("meth_algebra",                "think"),
    ("meth_calculus",               "think"),
    ("meth_geometry",               "think"),
    ("meth_statistics",             "think"),
    ("meth_logic",                  "think"),
    ("meth_visual",                 "visual"),
]

# Content that should never appear in a response (session logs, raw JSON, code bleed)
RESPONSE_REJECT = [
    lambda c: c.startswith("{") or c.startswith("[{"),      # raw JSON only
    lambda c: c.startswith("meth_"),                         # cell IDs
    lambda c: len(c.strip()) < 8,                            # too short
]


def is_valid_response_content(content):
    c = content.strip()
    return not any(check(c) for check in RESPONSE_REJECT)


# ── Domain routing ─────────────────────────────────────────────────────────────────

CODE_SIGNALS = {
    "code", "coding", "program", "programming", "function", "class",
    "import", "variable", "loop", "algorithm", "syntax", "compile", "debug",
    "python", "javascript", "java", "rust", "typescript", "html", "css",
    "react", "api", "database", "sql", "git", "docker", "bash",
    "script", "regex", "error", "exception", "stack", "array",
    "dict", "tuple", "string", "integer", "float", "boolean",
    "return", "print", "lambda", "async", "await", "implement", "build",
    "refactor", "method", "object", "interface", "module", "package",
    "framework", "library", "npm", "pip", "def", "self", "init",
}

VISUAL_SIGNALS = {
    "image", "picture", "photo", "visual", "see", "look", "colour", "color",
    "draw", "paint", "render", "display", "pixel", "resolution", "graphic",
    "design", "layout", "icon", "font", "animation", "diagram", "chart",
}

SPEAK_SIGNALS = {
    "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank", "please",
    "sorry", "greetings", "sup", "yo", "howdy", "cheers",
}

LISTEN_SIGNALS = {
    "listen", "hear", "audio", "sound", "voice", "speech", "tone",
    "pronunciation", "accent", "phonetic",
}


def route_query(query_text):
    """
    Determine which lobe(s) should fire alongside think.
    Think is always implicit — this returns the secondary lobes.
    Only the relevant domain fires; unrelated lobes stay silent.
    """
    words = set(query_text.lower().replace("?", " ").replace("!", " ")
                .replace(".", " ").replace(",", " ").replace(":", " ").split())

    relevant = set()

    if words & CODE_SIGNALS:
        relevant.add("code")
    if words & VISUAL_SIGNALS:
        relevant.add("visual")
    if words & SPEAK_SIGNALS:
        relevant.add("speak")
    if words & LISTEN_SIGNALS:
        relevant.add("listen")

    # Default: mind + speak for general knowledge / conversational queries
    if not relevant:
        relevant.add("mind")
        relevant.add("speak")

    return relevant


# ── Substrate helpers ──────────────────────────────────────────────────────────

def get_pool(substrate, lobe):
    prefixes = LOBE_PREFIXES[lobe]
    return [
        (cid, cell)
        for cid, cell in substrate.methodology_cells.items()
        if any(cid.startswith(p) for p in prefixes)
        # Never include console session cells in the access pool
        and not cid.startswith("meth_mind_console_")
    ]


def encode(substrate, text):
    v = np.array(substrate.hdc.encode(text), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def census(substrate):
    counts = {l: 0 for l in LOBE_PREFIXES}
    for cid in substrate.methodology_cells:
        if cid.startswith("meth_mind_console_"):
            continue
        for prefix, lobe in PREFIX_TO_LOBE:
            if cid.startswith(prefix):
                counts[lobe] += 1
                break
    return counts


# ── Response assembly ──────────────────────────────────────────────────────────

def assemble_response(lobe_results, substrate, q_vec, query_text, reader, room_hits=None):
    """
    Merge activated cells from all lobes into a unified pool, add any
    relevant room context, then let the FieldReader walk the substrate's
    activation field to produce the response.

    The field reader scores the full substrate (not just lobe slices),
    walks the connection graph from the most activated cell outward,
    and extracts the most query-relevant phrase from each walked cell.
    No templates. No phrase banks. The substrate speaks for itself.
    """
    # Merge all lobe hits — deduplicate by cell_id, keep highest sim
    seen_cids = {}
    for lobe, (hits, status) in lobe_results.items():
        for sim, cid, cell in hits:
            content = str(getattr(cell, "content", "")).strip()
            if not is_valid_response_content(content):
                continue
            if cid not in seen_cids or sim > seen_cids[cid][0]:
                seen_cids[cid] = (sim, cid, cell)

    activated_cells = sorted(seen_cids.values(), reverse=True)

    # Room context — inject prior session cells if meaningfully relevant
    if room_hits:
        for sim, cid, cell in room_hits[:2]:
            if sim > 0.70 and cid not in seen_cids:
                content = str(getattr(cell, "content", "")).strip()
                if is_valid_response_content(content):
                    activated_cells.append((sim, cid, cell))

    return reader.read(substrate, q_vec, query_text, activated_cells)


# ── Conversation flow wiring ───────────────────────────────────────────────────

def _wire_exchange(substrate, query, response):
    """
    Wire each live exchange as a connected pair so IA learns conversation
    flow from its own conversations in real time.
    query  → response : forward connection 0.7
    response → query  : reverse connection 0.3
    """
    from living_cell import LivingCell

    def _make(text, prefix):
        uid     = hashlib.md5(text.encode()).hexdigest()[:12]
        cell_id = f"{prefix}_{uid}"
        if cell_id in substrate.methodology_cells:
            return substrate.methodology_cells[cell_id]
        v    = np.array(substrate.hdc.encode(text), dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return None
        cell = LivingCell(cell_id, content=text, source_table="conv_flow",
                          confidence=1.1, source="conv_flow")
        cell.dna     = v / norm
        cell.anchor  = text[:200]
        cell.cell_id = cell_id
        substrate.methodology_cells[cell_id] = cell
        substrate._persist_methodology_cell(cell)
        return cell

    if len(query.strip()) < 4 or len(response.strip()) < 4:
        return

    cq = _make(query,    "meth_speak_conv")
    cr = _make(response, "meth_speak_conv")
    if cq is None or cr is None:
        return

    if not hasattr(cq, "connections") or cq.connections is None:
        cq.connections = {}
    if not hasattr(cr, "connections") or cr.connections is None:
        cr.connections = {}

    cq.connections[cr.cell_id] = max(cq.connections.get(cr.cell_id, 0.0), 0.7)
    cr.connections[cq.cell_id] = max(cr.connections.get(cq.cell_id, 0.0), 0.3)

    substrate._persist_methodology_cell(cq)
    substrate._persist_methodology_cell(cr)


# ── Session logging ────────────────────────────────────────────────────────────

def log_exchange(substrate, query, response, session_cells):
    """Log the exchange. These cells are NEVER added to the access pool."""
    from living_cell import LivingCell
    ts = int(time.time())
    uid = hashlib.md5(f"{ts}{query}".encode()).hexdigest()[:10]
    cell_id = f"meth_mind_console_{uid}"
    content = json.dumps({"q": query, "a": response[:300], "ts": ts})
    cell = LivingCell(cell_id, content=content,
                      source_table="console", confidence=1.1, source="console")
    cell.dna    = encode(substrate, query)
    cell.anchor = query[:200]
    substrate.methodology_cells[cell_id] = cell
    substrate._persist_methodology_cell(cell)
    session_cells.append((query, response))


# ── Display ────────────────────────────────────────────────────────────────────

LOBE_COL = {
    "speak": "\033[96m", "listen": "\033[94m", "mind":  "\033[95m",
    "think": "\033[93m", "code":   "\033[92m", "visual":"\033[91m",
}
STATUS_COL = {
    "found":    "\033[32m",   # green
    "created":  "\033[33m",   # yellow
    "partial":  "\033[34m",   # blue
    "gap":      "\033[31m",   # red
    "empty":    "\033[31m",   # red
    "learned":  "\033[96m",   # cyan  — new knowledge staged in think
    "staging":  "\033[36m",   # teal  — accessing staged knowledge, confidence building
    "promoted": "\033[35m",   # magenta — staged cell graduated to target lobe
    "deferred": "\033[90m",   # grey  — think resolved, lobe defers
}

def status_col(status):
    """Handle 'learned:source' compound status."""
    key = status.split(":")[0]
    return STATUS_COL.get(key, "")
R = "\033[0m"
B = "\033[1m"
D = "\033[2m"


def bar(v, w=18):
    f = int(v * w)
    return "█" * f + "░" * (w - f)


def print_activation(lobe_results):
    print(f"\n{D}  ┌─ cognitive access ──────────────────────────────────────────┐{R}")
    for lobe in ["think", "mind", "speak", "listen", "code", "visual"]:
        hits, status = lobe_results.get(lobe, ([], "empty"))
        if not hits:
            continue
        top_sim = hits[0][0]
        top_content = str(getattr(hits[0][2], "content", ""))[:45].replace("\n", " ")
        sc = status_col(status)
        lc = LOBE_COL.get(lobe, "")
        print(f"{D}  │{R} {lc}{lobe:<8}{R} {bar(top_sim)} {top_sim:.3f}  {sc}[{status}]{R}  {D}{top_content}…{R}")
    print(f"{D}  └───────────────────────────────────────────────────────────────┘{R}")


def print_banner():
    print(f"""
{B}╔══════════════════════════════════════════════════════════════╗
║                    IA CONSOLE  v2.0                          ║
║           Cognitive Access — Check, Correct, Create          ║
╚══════════════════════════════════════════════════════════════╝{R}
  {D}/census  /introspect  /lobe <name>  /cells <n>  /learn  /exit{R}
""")


# ── Main loop ──────────────────────────────────────────────────────────────────

def run():
    from system import AgenticSystem
    from cognitive_access import CognitiveAccess
    from session_room import Room
    from field_reader import FieldReader

    print("  Loading IA substrate...")
    substrate = AgenticSystem(hdc_dim=384, slim=True, skip_cells=True)
    print(f"  Substrate online — {len(substrate.methodology_cells):,} cells")

    archive = Room.load_archive()
    print(f"  Session archive — {len(archive)} prior session(s)\n")

    access = CognitiveAccess()
    room   = Room(substrate)
    reader = FieldReader()
    print_banner()

    session_cells = []
    last_results  = {}
    last_query    = ""

    while True:
        try:
            raw = input(f"{B}You ›{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  IA: Shutting down. {len(session_cells)} exchanges logged.")
            room.close()
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────────────

        if raw == "/exit":
            print("  IA: Goodbye.")
            room.close()
            break

        if raw == "/census":
            counts = census(substrate)
            total  = sum(counts.values())
            mx     = max(counts.values()) or 1
            print(f"\n  {B}Lobe Census{R} — {total:,} cells in memory")
            for lobe, count in sorted(counts.items(), key=lambda x: -x[1]):
                lc  = LOBE_COL.get(lobe, "")
                thr = "✓" if count >= 1000 else "⚠ below threshold"
                print(f"  {lc}{lobe:<8}{R} {bar(count/mx, 28)} {count:>5,}  {thr}")
            print()
            continue

        if raw == "/introspect":
            if not last_results:
                print("  No activation yet — send a message first.\n")
                continue
            print(f"\n  {B}Introspection — \"{last_query}\"{R}")
            for lobe in ["think", "mind", "speak", "listen", "code", "visual"]:
                hits, status = last_results.get(lobe, ([], "empty"))
                if not hits:
                    continue
                lc = LOBE_COL.get(lobe, "")
                sc = status_col(status)
                print(f"\n  {lc}{lobe.upper()}{R}  {sc}[{status}]{R}")
                for sim, cid, cell in hits[:3]:
                    content = str(getattr(cell, "content", "")).strip()[:120].replace("\n", " ")
                    print(f"    {sim:.3f}  {D}{cid}{R}")
                    print(f"           {content}")
            print()
            continue

        if raw.startswith("/lobe "):
            lobe = raw.split(None, 1)[1].strip().lower()
            if lobe not in LOBE_PREFIXES:
                print(f"  Unknown lobe. Choose: {', '.join(LOBE_PREFIXES)}\n")
                continue
            if not last_query:
                print("  Send a message first.\n")
                continue
            q_vec = encode(substrate, last_query)
            pool  = get_pool(substrate, lobe)
            hits, status = access.access(substrate, pool, q_vec, last_query, lobe)
            lc = LOBE_COL.get(lobe, "")
            sc = STATUS_COL.get(status, "")
            print(f"\n  {lc}{lobe.upper()}{R} {sc}[{status}]{R} — {len(pool):,} cells")
            for sim, cid, cell in hits[:6]:
                content = str(getattr(cell, "content", "")).strip()[:160].replace("\n", " ")
                print(f"  {sim:.3f}  {content}")
            print()
            continue

        if raw.startswith("/cells"):
            parts = raw.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            if not last_results:
                print("  No activation yet.\n")
                continue
            all_flat = []
            for lobe, (hits, status) in last_results.items():
                for sim, cid, cell in hits:
                    all_flat.append((sim, lobe, cid, cell))
            all_flat.sort(reverse=True)
            print(f"\n  {B}Top {n} activated cells{R}")
            for sim, lobe, cid, cell in all_flat[:n]:
                lc = LOBE_COL.get(lobe, "")
                content = str(getattr(cell, "content", "")).strip()[:200].replace("\n", " ")
                print(f"\n  {sim:.3f}  {lc}[{lobe}]{R}  {D}{cid}{R}")
                print(f"  {content}")
            print()
            continue

        if raw == "/learn":
            print(f"\n  {B}Session log{R} — {len(session_cells)} exchanges")
            for q, a in session_cells:
                print(f"  {D}Q:{R} {q[:80]}")
                print(f"  {D}A:{R} {a[:80]}")
                print()
            continue

        # ── Tool injection: @file and !command ───────────────────────────────

        injected = []
        words    = raw.split()
        clean_words = []
        for word in words:
            if word.startswith('@') and len(word) > 1:
                fpath = word[1:]
                # resolve relative to cwd or home
                if not os.path.isabs(fpath):
                    fpath = os.path.join(os.getcwd(), fpath)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath) as fh:
                            content = fh.read()
                        lines   = content.splitlines()
                        preview = '\n'.join(lines[:120])  # first 120 lines
                        injected.append(f"[file: {os.path.basename(fpath)}]\n{preview}")
                        print(f"  {D}↳ injected {os.path.basename(fpath)} ({len(lines)} lines){R}")
                    except Exception as e:
                        print(f"  {D}↳ could not read {fpath}: {e}{R}")
                else:
                    print(f"  {D}↳ file not found: {fpath}{R}")
                    clean_words.append(word)
            elif word.startswith('!') and len(word) > 1:
                cmd = word[1:] + ' ' + ' '.join(
                    w for w in words[words.index(word)+1:] if not w.startswith('@'))
                try:
                    import subprocess
                    result = subprocess.run(cmd, shell=True, capture_output=True,
                                            text=True, timeout=10)
                    out = (result.stdout + result.stderr).strip()[:2000]
                    injected.append(f"[command: {cmd}]\n{out}")
                    print(f"  {D}↳ ran: {cmd}{R}")
                except Exception as e:
                    print(f"  {D}↳ command failed: {e}{R}")
                break
            else:
                clean_words.append(word)

        # Rebuild query with injected context appended
        base_query = ' '.join(clean_words).strip() or raw
        if injected:
            raw = base_query + '\n\n' + '\n\n'.join(injected)

        # ── Cognitive access — think-first cascade ───────────────────────────

        last_query = base_query
        q_vec = encode(substrate, base_query)

        # Room check — what from this conversation is relevant right now?
        room_hits = room.query(q_vec, threshold=0.62)

        # Step 1: Think always fires first (unrestricted pool)
        think_pool = get_pool(substrate, "think")
        think_hits, think_status = access.access(
            substrate, think_pool, q_vec, raw, "think"
        )
        think_found = think_status == "found"

        # Step 2: Route to relevant lobe(s) only — skip unrelated lobes
        relevant = route_query(raw)

        lobe_results = {"think": (think_hits, think_status)}
        for lobe in relevant:
            pool = get_pool(substrate, lobe)
            hits, status = access.access(
                substrate, pool, q_vec, raw, lobe,
                think_resolved=think_found,
            )
            lobe_results[lobe] = (hits, status)

        last_results = lobe_results

        # Note activated cell IDs for the session snapshot
        activated_ids = [cid for _, (hits, _) in lobe_results.items() for _, cid, _ in hits]
        room.note_activated(activated_ids)

        # Assemble and display
        print_activation(lobe_results)
        response = assemble_response(lobe_results, substrate, q_vec, raw, reader, room_hits=room_hits)
        field_info = f"{D}  ↳ field: {reader.last_field_size:,} cells  walk: {reader.last_walk_len} steps{R}"
        print(f"\n{B}  IA ›{R} {response}\n")
        print(field_info)

        # Prior session detected (fires once, silently loads ambient context)
        if room._prior and not getattr(room, "_prior_announced", False):
            room._prior_announced = True
            n = getattr(room, "_prior_ambient_count", 0)
            ts = room._prior.get("ts_start", 0)
            import datetime
            date_str = datetime.datetime.fromtimestamp(ts).strftime("%d %b %Y") if ts else "?"
            print(f"  {D}↳ prior context surfacing — {n} cells from {date_str}{R}\n")

        # Surface knowledge acquisition events this turn
        events = []
        for lobe, (hits, status) in lobe_results.items():
            if status.startswith("learned"):
                source = status.split(":")[1] if ":" in status else "?"
                events.append(f"  {D}↳ [{lobe}] staged {len(hits)} new cells from {source} into think{R}")
            elif status == "staging":
                staged = [h for h in hits if getattr(h[2], "source", "") == "inquisition"]
                if staged:
                    avg_conf = sum(getattr(h[2], "confidence", 0) for h in staged) / len(staged)
                    events.append(f"  {D}↳ [{lobe}] staging — {len(staged)} cells, avg conf {avg_conf:.2f}{R}")
            elif status == "promoted":
                n = len([h for h in hits if getattr(h[2], "source", "") == "inquisition_promoted"])
                events.append(f"  \033[35m↳ [{lobe}] {n} cell(s) promoted — permanently learned{R}")
        for e in events:
            print(e)
        if events:
            print()

        # Curiosity — fire after any partial/correct cycle this turn
        new_curiosity = access.maybe_spark_curiosity(substrate)
        for cell in new_curiosity:
            print(f"  {D}↳ curiosity sparked: {cell.content[:90]}…{R}")
        if new_curiosity:
            print()

        # Log exchange, wire as connected pair (interactive sessions only), add to room
        log_exchange(substrate, raw, response, session_cells)
        if sys.stdin.isatty():
            _wire_exchange(substrate, raw, response)
        room.add_exchange(raw, response, q_vec)


if __name__ == "__main__":
    run()
