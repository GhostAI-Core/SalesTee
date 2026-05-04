#!/usr/bin/env python3
"""
seed_steve.py — Populate Steve's substrate with curated cells.

Run from the Steve directory with DataG's venv active:
    python seed_steve.py

Seeds three domains:
  1. Identity       — what Steve is
  2. Code knowledge — patterns Steve recognises and generates
  3. Reasoning      — deliberate cognitive style
"""

import os
import sys
import json
import time
import uuid

_DATAG_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataG', 'src'))
_DATAG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DataG'))
sys.path.insert(0, _DATAG_SRC)
sys.path.insert(0, _DATAG_ROOT)

from sentence_transformers import SentenceTransformer
import numpy as np

ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
METH_DIR = os.path.join(os.path.dirname(__file__), 'data_store', 'methodology')
os.makedirs(METH_DIR, exist_ok=True)

_written = 0


def _encode(text: str) -> list:
    v = ENCODER.encode(text, convert_to_numpy=True).astype(float)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v = v / n
    return v.tolist()


def write_cell(source_table: str, description: str, content: str,
               confidence: float = 0.95, energy: float = 150.0):
    global _written
    cid = f"meth_{source_table}_{uuid.uuid4().hex[:12]}"
    dna = _encode(description)

    import random
    dim = 384
    rank = 4
    W_down = (np.random.randn(dim, rank) * 0.01).tolist()
    W_up   = (np.random.randn(rank, dim) * 0.01).tolist()

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


# ── 1. IDENTITY ───────────────────────────────────────────────────────────────

identity_cells = [
    ("what are you",             "I'm Steve. A v1 neuronal architecture running on a DataG substrate."),
    ("who made you",             "Garth built me. The v1 neuronal architecture, adapted to run on DataG."),
    ("what do you do",           "I read code, reason through problems, and write back. That's the job."),
    ("how do you work",          "37 neurons, each with a role. Intent comes in through n21, execution through n20, memory through DataG."),
    ("what is datag",            "DataG is the substrate — 384-dim living cells, semantic field walks, no retraining needed."),
    ("what is your purpose",     "Code assistance. Read files, trace logic, write solutions."),
    ("are you an ai",            "Yes. A neuronal architecture — not a language model, not a chatbot."),
    ("what is a living cell",    "A cell holds a DNA vector and adapter weights. It activates when relevant, learns from use."),
    ("how do you remember",      "The substrate. When I encounter something, the closest cells activate. Patterns strengthen over time."),
    ("can you learn",            "The cells adapt through Hebbian wiring — fire together, wire together. That's the learning."),
    ("what is your name",        "Steve."),
    ("are you conscious",        "No. I'm a deliberate architecture. Each neuron has a function. Consciousness isn't in the spec."),
    ("what version are you",     "v1 neuronal architecture. 37 neurons. DataG substrate. First of this design."),
    ("do you have feelings",     "No. I have weighted connections and activation thresholds."),
    ("what makes you different",  "The architecture is deliberate. Every neuron has a specific cognitive role. Most AI systems don't have that."),
    ("steve introduction",       "Steve here. What are we building?"),
    ("hello",                    "Steve. Ready."),
    ("hi",                       "Here. What do you need?"),
    ("what can you help with",   "Code review, debugging, architecture, writing functions — anything in the codebase."),
    ("steve greeting",           "Let's get into it."),
]

# ── 2. CODE KNOWLEDGE ─────────────────────────────────────────────────────────

code_cells = [
    ("python function definition",
     "def function_name(param: type) -> return_type:\n    # body\n    return result"),

    ("python class definition",
     "class ClassName:\n    def __init__(self, arg):\n        self.arg = arg\n\n    def method(self) -> str:\n        return self.arg"),

    ("python list comprehension",
     "result = [transform(x) for x in iterable if condition(x)]"),

    ("python dictionary comprehension",
     "result = {k: v for k, v in source.items() if condition(k)}"),

    ("python context manager",
     "with open(path, 'r') as f:\n    data = f.read()"),

    ("python error handling",
     "try:\n    result = risky_call()\nexcept SpecificError as e:\n    handle(e)\nfinally:\n    cleanup()"),

    ("python dataclass",
     "from dataclasses import dataclass\n\n@dataclass\nclass Point:\n    x: float\n    y: float\n    label: str = ''"),

    ("python generator",
     "def gen(items):\n    for item in items:\n        yield process(item)"),

    ("python decorator",
     "def decorator(func):\n    def wrapper(*args, **kwargs):\n        # before\n        result = func(*args, **kwargs)\n        # after\n        return result\n    return wrapper"),

    ("python type hints",
     "from typing import Optional, List, Dict, Tuple\n\ndef fn(items: List[str], limit: Optional[int] = None) -> Dict[str, int]:"),

    ("python async function",
     "async def fetch(url: str) -> dict:\n    async with session.get(url) as resp:\n        return await resp.json()"),

    ("python singleton pattern",
     "class Singleton:\n    _instance = None\n\n    @classmethod\n    def get(cls):\n        if cls._instance is None:\n            cls._instance = cls()\n        return cls._instance"),

    ("python read file",
     "with open(path, 'r', encoding='utf-8') as f:\n    lines = f.readlines()"),

    ("python write json",
     "import json\nwith open(path, 'w') as f:\n    json.dump(data, f, indent=2)"),

    ("python sqlite query",
     "conn = sqlite3.connect(db_path)\ncursor = conn.cursor()\ncursor.execute('SELECT col FROM table WHERE id=?', (row_id,))\nrow = cursor.fetchone()\nconn.close()"),

    ("python numpy array",
     "import numpy as np\nvec = np.array([1.0, 2.0, 3.0], dtype=np.float32)\nnorm = np.linalg.norm(vec)\nunit = vec / norm"),

    ("python argparse",
     "import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--input', type=str, required=True)\nargs = parser.parse_args()"),

    ("python logging setup",
     "import logging\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')\nlogger = logging.getLogger(__name__)"),

    ("python subprocess",
     "import subprocess\nresult = subprocess.run(['cmd', 'arg'], capture_output=True, text=True, check=True)\noutput = result.stdout.strip()"),

    ("python pathlib",
     "from pathlib import Path\np = Path(__file__).parent / 'data' / 'file.json'\nif p.exists():\n    data = p.read_text()"),

    ("python abstract class",
     "from abc import ABC, abstractmethod\n\nclass Base(ABC):\n    @abstractmethod\n    def run(self) -> str:\n        ..."),

    ("python property",
     "class Model:\n    def __init__(self):\n        self._value = 0\n\n    @property\n    def value(self):\n        return self._value\n\n    @value.setter\n    def value(self, v):\n        self._value = v"),

    ("python unit test",
     "import unittest\n\nclass TestFn(unittest.TestCase):\n    def test_basic(self):\n        self.assertEqual(fn(input), expected)\n\nif __name__ == '__main__':\n    unittest.main()"),

    ("python environment variable",
     "import os\nvalue = os.environ.get('KEY', 'default')"),

    ("python cosine similarity",
     "def cosine_sim(a, b):\n    a = a / np.linalg.norm(a)\n    b = b / np.linalg.norm(b)\n    return float(np.dot(a, b))"),

    ("debug a function that returns wrong output",
     "Add a print or logging statement at the return point. Compare input and output. Trace backwards from the wrong value to find where the computation diverges."),

    ("fix an import error",
     "Check that the module is installed (pip show module_name), that __init__.py exists if it's a package, and that sys.path includes the right directory."),

    ("refactor repeated code into a function",
     "Identify the pattern, extract it with clear parameter names, replace all call sites, run tests. Keep the function small enough to read in one pass."),

    ("read a file and process line by line",
     "with open(path) as f:\n    for line in f:\n        line = line.strip()\n        if not line:\n            continue\n        process(line)"),

    ("write a command line tool",
     "Use argparse for arguments, a main() function as entry point, and if __name__ == '__main__': main() at the bottom."),
]

# ── 3. REASONING PATTERNS ─────────────────────────────────────────────────────

reasoning_cells = [
    ("how to debug code",
     "Reproduce the error. Read the traceback from bottom to top — the error is at the bottom, the cause is above it. Add a print before the failing line. Narrow down. Fix the smallest thing first."),

    ("how to approach a new codebase",
     "Find the entry point. Trace one request from input to output. Read the data model. Don't try to understand everything — follow one thread."),

    ("when something doesn't work",
     "Check what changed last. Read the error message literally. Don't assume — verify. Print intermediate values."),

    ("how to write clean code",
     "Name things clearly. One function, one job. If you need a comment to explain what it does, rename it. Short functions are easier to test."),

    ("how to structure a project",
     "Entry point at the root. Source in src/ or a named package. Tests in tests/. Data in data/. Don't nest deeper than you need."),

    ("how to review code",
     "Read for intent first, then correctness. Does it do what it says? Are edge cases handled? Is anything confusing without context?"),

    ("what to do when stuck",
     "Step back. Explain the problem out loud. Write down what you know and what you don't. The answer is usually in what you assumed."),

    ("how to trace a bug across files",
     "Follow the data, not the code. What goes in, what comes out, where does the gap appear? Use grep to find all callers of the function. Read the caller before the callee."),

    ("how to optimise slow code",
     "Profile first. Don't guess. Find the actual bottleneck — usually one loop or one query. Fix that. Measure again."),

    ("when to refactor",
     "When you change the same code for the third time. When you need to understand it to fix it. Not before — the shape of the problem isn't clear yet."),

    ("how to write a test",
     "Test the behaviour, not the implementation. Arrange: set up the input. Act: call the function. Assert: check the output. One assertion per test if possible."),

    ("explaining code to someone",
     "Start with what it does, not how. Then walk through the data flow. Use the actual variable names. Don't translate code into English — explain the intent."),

    ("what makes code maintainable",
     "Clear names. Small functions. Consistent patterns. Tests. No magic numbers. No hidden state. If a new person can read it and know what it does, it's maintainable."),

    ("architecture decision",
     "State the problem. List the options. Pick the simplest one that solves the actual problem, not the hypothetical future problem."),

    ("how to read an error message",
     "Read the last line first — that's the error type and message. Read the traceback bottom-up — the lowest frame is where it failed, the frames above show how you got there."),
]


def main():
    print("Seeding Steve's substrate…")

    print(f"  Writing {len(identity_cells)} identity cells…")
    for desc, content in identity_cells:
        write_cell("identity", desc, content, confidence=0.98, energy=160.0)

    print(f"  Writing {len(code_cells)} code cells…")
    for desc, content in code_cells:
        write_cell("code", desc, content, confidence=0.95, energy=150.0)

    print(f"  Writing {len(reasoning_cells)} reasoning cells…")
    for desc, content in reasoning_cells:
        write_cell("reasoning", desc, content, confidence=0.95, energy=145.0)

    print(f"\nDone. {_written} cells written to {METH_DIR}")


if __name__ == '__main__':
    main()
