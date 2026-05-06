#!/usr/bin/env python3
"""
ingest_doc.py — Feed documents into Tee's memory.

Supports: PDF, DOCX, TXT, MD, or paste text directly.
Walks you through chunking, reviewing, and saving cells.

Usage:
    python ingest_doc.py
    python ingest_doc.py path/to/file.pdf
"""

import os
import sys
import textwrap
import argparse

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

W = 72

# source_table, display label, explanation of where it goes
TYPES = [
    ('__product__',     'Product / service',
     'Tee sells from this. Kept word-for-word — exact pricing, features,\n'
     '     steps. Tee will NOT paraphrase this content.'),

    ('identity',        'Sales identity / persona',
     'Shapes HOW Tee speaks — her tone, confidence, values. The decoder\n'
     '     uses this to generate responses in Tee\'s voice.'),

    ('reasoning',       'Objection handling',
     'Counter-arguments and persuasion logic. Tee draws on these when\n'
     '     a prospect pushes back or stalls.'),

    ('conversation',    'General knowledge / background',
     'Context that Tee knows but isn\'t directly selling. Industry facts,\n'
     '     company background, supporting detail.'),

    ('__custom__',      'Something else — I\'ll name it myself',
     'You define the category. Treated like product content — Tee returns\n'
     '     it word-for-word without paraphrasing.'),
]


# ── Terminal helpers ──────────────────────────────────────────────────────────

def hr():
    print('─' * W)

def banner(text):
    print(f"\n{'='*W}")
    print(f"  {text}")
    print(f"{'='*W}\n")

def ask(prompt, options=None):
    while True:
        raw = input(f"  {prompt} ").strip()
        if raw:
            if options and raw not in options:
                print(f"  Please enter one of: {', '.join(options)}")
                continue
            return raw
        print("  (Please enter a value)")

def ask_yn(prompt) -> bool:
    ans = ask(f"{prompt} (yes / no)", options=['yes', 'no', 'y', 'n'])
    return ans.lower() in ('yes', 'y')

def wrap(text, indent=4):
    prefix = ' ' * indent
    return textwrap.fill(text, width=W - indent, initial_indent=prefix,
                         subsequent_indent=prefix)

def slugify(name: str) -> str:
    """Turn a product name into a safe source_table suffix: 'VOXI Pro' -> 'voxi_pro'"""
    import re
    return re.sub(r'[^a-z0-9]+', '_', name.lower().strip()).strip('_')


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_pdf(path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        print("\n  ERROR: pdfplumber is not installed.")
        print("  Run:  pip install pdfplumber\n")
        sys.exit(1)
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return '\n\n'.join(pages)

def extract_docx(path: str) -> str:
    try:
        from docx import Document
    except ImportError:
        print("\n  ERROR: python-docx is not installed.")
        print("  Run:  pip install python-docx\n")
        sys.exit(1)
    doc = Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return '\n\n'.join(paragraphs)

def extract_text(path: str) -> str:
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()

def load_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return extract_pdf(path)
    elif ext == '.docx':
        return extract_docx(path)
    elif ext in ('.txt', '.md', '.markdown', ''):
        return extract_text(path)
    else:
        print(f"\n  Unsupported file type: {ext}")
        print(  "  Supported: .pdf  .docx  .txt  .md\n")
        sys.exit(1)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, min_words: int = 10) -> list:
    raw_chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
    merged, carry = [], ''
    for chunk in raw_chunks:
        combined = (carry + ' ' + chunk).strip() if carry else chunk
        if len(combined.split()) < min_words:
            carry = combined
        else:
            merged.append(combined)
            carry = ''
    if carry:
        merged.append(carry)
    return merged

def ask_description(chunk: str) -> str:
    """
    Ask the user to write the retrieval description for this chunk.
    This IS the DNA — it determines what prospect questions hit this cell.
    """
    print(f"\n  Content preview:")
    print(wrap(chunk[:300] + ('...' if len(chunk) > 300 else ''), indent=4))
    print()
    print("  Write the question a prospect would ask to need this answer.")
    print("  Be specific. Examples:")
    print("    'can voxi book meetings automatically'")
    print("    'what happens when voxi cant handle a call'")
    print("    'how much does voxi cost'")
    print()
    while True:
        desc = input("  Description: ").strip()
        if len(desc) >= 5:
            return desc
        print("  (Too short — write the prospect question this cell answers)")


# ── Existing products in substrate ────────────────────────────────────────────

def list_known_products(meth_dir: str) -> list:
    """Return sorted list of product namespaces already in the substrate."""
    import json
    known = set()
    for fname in os.listdir(meth_dir):
        if not fname.endswith('.json'):
            continue
        try:
            with open(os.path.join(meth_dir, fname)) as f:
                d = json.load(f)
            t = d.get('source_table', '')
            if t.startswith('meth_product_'):
                known.add(t[len('meth_product_'):])
        except Exception:
            pass
    return sorted(known)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('file', nargs='?', default=None)
    args = parser.parse_args()

    banner("INGEST — Feed Documents into Tee's Memory")

    # ── Step 1: Get the text ─────────────────────────────────────────────────
    if args.file:
        path = args.file
        if not os.path.exists(path):
            print(f"  File not found: {path}\n")
            sys.exit(1)
        print(f"  Loading: {path}")
        raw_text = load_file(path)
    else:
        print("  No file given. Options:\n")
        print("    1. I have a file  (PDF, DOCX, TXT, MD)")
        print("    2. I want to paste text directly\n")
        choice = ask("Enter 1 or 2:", options=['1', '2'])
        if choice == '1':
            path = ask("File path:")
            if not os.path.exists(path):
                print(f"\n  File not found: {path}\n")
                sys.exit(1)
            raw_text = load_file(path)
        else:
            print("\n  Paste your text below. Type END on its own line when done.\n")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == 'END':
                    break
                lines.append(line)
            raw_text = '\n'.join(lines)

    word_count = len(raw_text.split())
    chunks = chunk_text(raw_text)
    print(f"\n  Found {word_count:,} words across {len(chunks)} sections.\n")

    # ── Step 2: Content type ─────────────────────────────────────────────────
    hr()
    print("\n  What type of content is this?\n")
    for i, (_, label, description) in enumerate(TYPES, 1):
        print(f"    {i}. {label}")
        print(f"       {description}\n")

    valid = [str(i) for i in range(1, len(TYPES) + 1)]
    type_choice = int(ask(f"Enter 1-{len(TYPES)}:", options=valid)) - 1
    source_key, type_label, _ = TYPES[type_choice]

    # ── Step 3: Resolve source_table ─────────────────────────────────────────
    meth_dir = os.path.join(os.path.dirname(__file__), 'data_store', 'methodology')

    if source_key == '__product__':
        known = list_known_products(meth_dir)
        print(f"\n  What product or service is this content about?\n")
        if known:
            print("  Existing products in Tee's memory:")
            for p in known:
                print(f"    - {p}")
            print()
            print("  Enter the product name exactly to add to an existing one,")
            print("  or type a new name to create a new product category.\n")
        else:
            print("  No products in Tee's memory yet. Enter the product name.\n")
            print("  Examples: VOXI, Acme CRM, ProSuite\n")
        product_name = ask("Product name:")
        product_slug = slugify(product_name)
        source_table = f'product_{product_slug}'
        print(f"\n  Got it — filing under: {source_table}\n")

    elif source_key == '__custom__':
        print(f"\n  What do you want to call this category?\n")
        print("  Use a short descriptive name. Examples: competitor_intel,")
        print("  case_studies, testimonials, faq_returns\n")
        custom_name = ask("Category name:")
        source_table = slugify(custom_name)
        print(f"\n  Got it — filing under: {source_table}")
        print("  (Tee will return this content word-for-word, like product content.)\n")

    else:
        source_table = source_key
        print(f"\n  Got it — tagging as: {type_label}\n")

    # ── Step 4: Scope ────────────────────────────────────────────────────────
    hr()
    print("\n  How much of this document do you want?\n")
    print("    1. Use everything")
    print("    2. Review each section — I'll decide what to keep")
    print("    3. Condense each section — tighten the language before saving\n")
    scope = ask("Enter 1-3:", options=['1', '2', '3'])

    # ── Step 5: Review chunks ────────────────────────────────────────────────
    approved = []

    if scope == '1':
        hr()
        print(f"\n  Saving all {len(chunks)} sections.")
        print("  For each one, write the question a prospect would ask to need this answer.\n")
        for i, chunk in enumerate(chunks, 1):
            hr()
            print(f"\n  Section {i} of {len(chunks)}:")
            desc = ask_description(chunk)
            approved.append((desc, chunk))
        print(f"\n  {len(approved)} sections labelled.\n")

    elif scope == '2':
        hr()
        print(f"\n  Reviewing {len(chunks)} sections. For each one:\n")
        print("    yes   — keep it as-is")
        print("    no    — skip it")
        print("    edit  — retype the content before saving\n")

        for i, chunk in enumerate(chunks, 1):
            hr()
            print(f"\n  Section {i} of {len(chunks)}:\n")
            print(wrap(chunk[:500] + ('...' if len(chunk) > 500 else ''), indent=4))
            print()
            action = ask("Keep this?", options=['yes', 'no', 'edit', 'y', 'n'])
            if action in ('no', 'n'):
                continue
            elif action == 'edit':
                print("\n  Type your replacement. Press Enter on a blank line when done:\n")
                lines = []
                while True:
                    line = input("    ")
                    if line == '':
                        break
                    lines.append(line)
                chunk = ' '.join(lines).strip()
                if not chunk:
                    print("  (Empty — skipped)")
                    continue
            desc = ask_description(chunk)
            approved.append((desc, chunk))

        print(f"\n  {len(approved)} of {len(chunks)} sections kept.\n")

    elif scope == '3':
        hr()
        print(f"\n  Condensing {len(chunks)} sections.\n")
        print("  I'll show you each one. Retype a tighter version, or press")
        print("  Enter to keep the original as-is.\n")

        for i, chunk in enumerate(chunks, 1):
            hr()
            print(f"\n  Section {i} of {len(chunks)} — original:\n")
            print(wrap(chunk[:500] + ('...' if len(chunk) > 500 else ''), indent=4))
            print()
            print("  Your tighter version (or press Enter to keep):")
            replacement = input("    ").strip()
            final = replacement if replacement else chunk
            desc = ask_description(final)
            approved.append((desc, final))

        print(f"\n  {len(approved)} sections ready.\n")

    if not approved:
        print("  Nothing to save. Exiting.\n")
        sys.exit(0)

    # ── Step 6: Confirm and save ─────────────────────────────────────────────
    hr()
    print(f"\n  Ready to save {len(approved)} cells to Tee's memory")
    print(f"  Category: {source_table}\n")
    for i, (desc, _) in enumerate(approved, 1):
        print(f"    {i}. {desc[:65]}")
    print()

    if not ask_yn("Confirm?"):
        print("\n  Cancelled. Nothing saved.\n")
        sys.exit(0)

    print("\n  [Loading Tee...]", flush=True)
    from datag_bridge import DataGBridge
    bridge = DataGBridge.get()
    if not bridge.ready:
        print("  ERROR: Tee's substrate failed to load.\n")
        sys.exit(1)

    # Cells from custom types and product namespaces use raw-content routing
    # (same as 'product'). The decoder was only trained on identity/reasoning.
    is_decoder_type = source_table in ('identity', 'reasoning', 'conversation')
    effective_table = source_table if is_decoder_type else f'product_{source_table}' if source_key == '__custom__' else source_table

    print()
    saved = 0
    for desc, content in approved:
        cid = bridge.save_cell(
            description=desc,
            content=content,
            source_table=effective_table,
            confidence=0.95,
            energy=150.0,
        )
        print(f"  + {cid[:42]}  \"{desc[:40]}\"")
        saved += 1

    hr()
    print(f"\n  Done. {saved} cells added to Tee's memory.")
    print(f"  Total cells now: {len(bridge.substrate.methodology_cells)}")
    print(f"\n  Tee can use this material immediately.")
    print(f"  Retrain the encoder after a significant batch of new content")
    print(f"  for best retrieval accuracy.\n")
    print('=' * W + '\n')


if __name__ == '__main__':
    main()
