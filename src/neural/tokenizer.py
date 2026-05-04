"""
Training Tee Tokenizer — word-level, built from Training Tee's own corpus.
No external dependencies. Handles both prose and code vocabulary.
"""

import re
import json
from pathlib import Path

PAD   = '<PAD>'
UNK   = '<UNK>'
BOS   = '<BOS>'
EOS   = '<EOS>'
SPEC  = [PAD, UNK, BOS, EOS]


_NO_SPACE_BEFORE = set('.,:;!?\')]}')
_NO_SPACE_AFTER  = set("'([{")
_CONTRACTIONS    = {"i ' m": "I'm", "don ' t": "don't", "it ' s": "it's",
                    "can ' t": "can't", "won ' t": "won't", "isn ' t": "isn't",
                    "aren ' t": "aren't", "that ' s": "that's", "there ' s": "there's"}

# Words that must never be auto-capitalised (Python keywords, etc.)
_NO_CAP = frozenset({
    'def', 'class', 'return', 'import', 'from', 'if', 'else', 'elif',
    'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'in',
    'and', 'or', 'not', 'is', 'none', 'true', 'false', 'self',
    'pass', 'break', 'continue', 'yield', 'lambda', 'async', 'await',
    'raise', 'del', 'assert', 'global', 'nonlocal',
})

# Words that always appear in a specific case
_FIXED_CASE = {
    'training_tee': 'Training Tee', 'datag': 'DataG', 'datag.': 'DataG.',
    'v1': 'v1', 'hdc': 'HDC', 'bos': 'BOS', 'eos': 'EOS',
}


def _detokenize(tokens: list[str]) -> str:
    """Rejoin tokens with proper spacing and capitalisation."""
    if not tokens:
        return ''

    # Rebuild string with punctuation rules
    out = ''
    for i, tok in enumerate(tokens):
        if i == 0:
            out = tok
        elif tok in _NO_SPACE_BEFORE or (out and out[-1] in _NO_SPACE_AFTER):
            out += tok
        else:
            out += ' ' + tok

    # Collapse hyphen gaps: "384 - dim" → "384-dim"
    import re as _re
    out = _re.sub(r'\s*-\s*', '-', out)

    # Fix contractions
    for raw, fixed in _CONTRACTIONS.items():
        out = out.replace(raw, fixed)

    # Capitalise first word of each sentence; respect _NO_CAP and _FIXED_CASE
    words   = out.split(' ')
    cap_next = True
    result  = []
    for w in words:
        lower = w.lower().rstrip('.,;:!?')
        if lower in _FIXED_CASE:
            suffix = w[len(lower):]
            result.append(_FIXED_CASE[lower] + suffix)
        elif cap_next and lower not in _NO_CAP and w and w[0].isalpha():
            result.append(w[0].upper() + w[1:])
        else:
            result.append(w)
        # Always consume cap_next after any word — prevents cascade to second word
        cap_next = False
        if w.endswith(('.', '!', '?')):
            cap_next = True
    return ' '.join(result)


def _tokenize(text: str) -> list[str]:
    """Split text into tokens — handles prose, Python code, and punctuation."""
    # Normalize whitespace
    text = text.strip()
    # Split on word boundaries but keep useful symbols together
    tokens = re.findall(
        r'\w+|'                          # words / identifiers / numbers
        r'[+\-*/=<>!&|^~%@]+|'           # operators
        r'[(){}[\],.;:\'"\\]',           # punctuation / brackets
        text
    )
    return [t.lower() for t in tokens if t.strip()]


class TrainingTeeTokenizer:
    """
    Word-level tokenizer built from Training Tee's seed corpus.
    Vocab is frozen after build — add cells, then rebuild.
    """

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._built = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, texts: list[str], min_freq: int = 1) -> 'TrainingTeeTokenizer':
        """Build vocab from a list of texts."""
        freq: dict[str, int] = {}
        for text in texts:
            for tok in _tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1

        vocab = SPEC + sorted(t for t, c in freq.items() if c >= min_freq)
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._built = True
        return self

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, text: str, max_len: int = 128,
               add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Text → token IDs (with BOS/EOS). Truncates to max_len."""
        unk  = self.token2id[UNK]
        ids  = [self.token2id.get(t, unk) for t in _tokenize(text)]
        if add_bos:
            ids = [self.token2id[BOS]] + ids
        if add_eos:
            ids = ids + [self.token2id[EOS]]
        return ids[:max_len]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Token IDs → text string, with punctuation spacing fixed."""
        special = set(SPEC)
        tokens = [self.id2token.get(i, UNK) for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in special]
        return _detokenize(tokens)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'token2id': self.token2id}, f)

    @classmethod
    def load(cls, path: str) -> 'TrainingTeeTokenizer':
        tok = cls()
        with open(path) as f:
            data = json.load(f)
        tok.token2id = data['token2id']
        tok.id2token = {int(i) if isinstance(i, str) else i: t
                        for t, i in tok.token2id.items()}
        tok._built = True
        return tok

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token2id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token2id[EOS]

    def __len__(self) -> int:
        return self.vocab_size
