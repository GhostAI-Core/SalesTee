"""
Codebase Seeder
===============
Parses DataG and IA_code_base Python source files into function/class chunks.

Each chunk becomes a cell where:
  - DNA  = encoded from a natural language DESCRIPTION of what the code does
  - content = the actual code itself

This teaches the decoder the pattern: semantic intent → code output.
The geometry handles mode switching — code queries produce conjugate vectors
in code space, so the decoder outputs code. Conversational queries stay in prose.

Run: python seed_codebase.py
"""

import ast
import hashlib
import json
import os
import time
import numpy as np

_HERE     = os.path.dirname(os.path.abspath(__file__))
CELLS_DIR = os.path.join(_HERE, 'data_store', 'methodology')
DNA_DIM   = 384

SOURCE_ROOTS = [
    os.path.join(_HERE, 'src'),
    os.path.join(_HERE, '..', 'IA_code_base', 'src'),
    os.path.join(_HERE, '..', 'IA_code_base'),
]

SKIP_DIRS  = {'__pycache__', '.venv', 'venv', '.git', 'node_modules', 'migrations'}
SKIP_FILES = {'__init__.py'}
MIN_LINES  = 4    # skip trivial one-liners
MAX_LINES  = 120  # skip massive monoliths (too noisy as a single cell)


# ── AST parsing ───────────────────────────────────────────────────────────────

def _node_source(source_lines: list[str], node) -> str:
    start = node.lineno - 1
    end   = getattr(node, 'end_lineno', start + 1)
    return '\n'.join(source_lines[start:end])


def _describe_function(node) -> str:
    """Build a natural language description from signature + docstring."""
    docstring = ast.get_docstring(node) or ''
    args = [a.arg for a in node.args.args if a.arg != 'self']
    sig  = f"{node.name}({', '.join(args)})"
    kind = 'async function' if isinstance(node, ast.AsyncFunctionDef) else 'function'

    if docstring:
        doc = docstring.split('\n')[0].strip()
        return f"Python {kind} {sig}: {doc}"
    return f"Python {kind} {sig}"


def _describe_class(node) -> str:
    docstring = ast.get_docstring(node) or ''
    if docstring:
        doc = docstring.split('\n')[0].strip()
        return f"Python class {node.name}: {doc}"
    return f"Python class {node.name}"


def extract_chunks(filepath: str) -> list[tuple[str, str]]:
    """Return (description, code) pairs from a Python file."""
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            source = f.read()
    except Exception:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    chunks = []
    seen   = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not hasattr(node, 'end_lineno'):
            continue

        code  = _node_source(source_lines, node)
        n_lines = code.count('\n') + 1

        if n_lines < MIN_LINES or n_lines > MAX_LINES:
            continue

        key = hashlib.md5(code.encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)

        if isinstance(node, ast.ClassDef):
            desc = _describe_class(node)
        else:
            desc = _describe_function(node)

        chunks.append((desc, code))

    return chunks


def walk_sources() -> list[tuple[str, str, str]]:
    """Yield (filepath, description, code) across all source roots."""
    results = []
    seen_hashes = set()

    for root in SOURCE_ROOTS:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if not fname.endswith('.py') or fname in SKIP_FILES:
                    continue
                fpath = os.path.join(dirpath, fname)
                for desc, code in extract_chunks(fpath):
                    h = hashlib.md5(code.encode()).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    results.append((fpath, desc, code))

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading encoder…")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    chunks = walk_sources()
    print(f"Found {len(chunks):,} code chunks across source roots\n")

    os.makedirs(CELLS_DIR, exist_ok=True)
    written = 0
    skipped = 0
    errors  = 0

    for fpath, desc, code in chunks:
        uid  = hashlib.md5(code.encode()).hexdigest()[:12]
        cid  = f"meth_code_src_{uid}"
        path = os.path.join(CELLS_DIR, f"{cid}.json")

        if os.path.exists(path):
            skipped += 1
            continue

        # DNA encoded from description (natural language intent)
        # Content stored is the actual code (what decoder learns to output)
        try:
            vec  = encoder.encode(desc, convert_to_numpy=True).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                errors += 1
                continue
            vec = vec / norm
        except Exception:
            errors += 1
            continue

        W_down = (np.random.randn(DNA_DIM, 4) * 0.01).astype(np.float32)
        W_up   = (np.random.randn(4, DNA_DIM) * 0.01).astype(np.float32)

        rel_path = os.path.relpath(fpath, _HERE)
        cell = {
            "id":               cid,
            "content":          code,
            "source_table":     "codebase",
            "confidence":       0.90,
            "source":           "codebase_parse",
            "dna":              vec.tolist(),
            "W_down":           W_down.tolist(),
            "W_up":             W_up.tolist(),
            "connections":      {},
            "energy":           130.0,
            "activation_count": 0,
            "last_activated":   0,
            "birth_time":       time.time(),
            "meta":             {
                "description": desc,
                "file":        rel_path,
                "domain":      "code",
            },
        }

        try:
            with open(path, 'w') as f:
                json.dump(cell, f)
            written += 1
            if written % 50 == 0 or written <= 10:
                print(f"  [{written:4d}] {desc[:72]}")
        except Exception:
            errors += 1

    print(f"\n  Done.")
    print(f"  Written:  {written:,}")
    print(f"  Skipped:  {skipped:,} (already exist)")
    print(f"  Errors:   {errors:,}")
    print(f"  Restart DataG to load the new code cells.")


if __name__ == '__main__':
    main()
