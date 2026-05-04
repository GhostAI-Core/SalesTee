"""
IA Vocabulary — Built from Dictionary DNA
==========================================
Extracts vocabulary from IA's training sentences.
Initialises token embeddings from dictionary cell DNA vectors so every
word starts with the semantic position it already holds in DataG's space.

Words not found in the dictionary get small random embeddings and learn
their position from context during training.
"""

import json
import os
import re
import glob
import numpy as np
from collections import Counter


PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIAL_TOKENS = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']

DNA_DIM = 384


class IAVocabulary:

    def __init__(self):
        self.word2idx: dict = {}
        self.idx2word: list = []
        self.embeddings: np.ndarray = None   # (vocab_size, DNA_DIM)

    # ── Tokenisation ──────────────────────────────────────────────────────────

    @staticmethod
    def tokenise(text: str) -> list[str]:
        """Word + punctuation level tokenisation — lowercase."""
        return re.findall(r"[a-zA-Z']+|[.,!?;:\-]", text.lower())

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenise(text)
        return [BOS_IDX] + [self.word2idx.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]

    def decode(self, indices: list[int]) -> str:
        words = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, BOS_IDX):
                continue
            token = self.idx2word[idx] if idx < len(self.idx2word) else '[UNK]'
            words.append(token)
        text = ' '.join(words)
        # Close space before punctuation
        text = re.sub(r" ([.,!?;:\-])", r'\1', text)
        # Fix "i'" → "I'" at sentence start
        text = re.sub(r"^i\b", "I", text)
        text = re.sub(r" i'", " I'", text)
        if text:
            text = text[0].upper() + text[1:]
        return text

    def __len__(self):
        return len(self.idx2word)

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, sentences: list[str], dict_cell_dir: str,
              min_freq: int = 1) -> 'IAVocabulary':
        """
        Build vocab from training sentences.
        Initialise embeddings from dictionary cell DNA where available.
        """
        # Count all tokens
        counts = Counter()
        for s in sentences:
            counts.update(self.tokenise(s))

        words = [w for w, c in counts.most_common() if c >= min_freq]
        all_tokens = SPECIAL_TOKENS + words
        self.word2idx = {w: i for i, w in enumerate(all_tokens)}
        self.idx2word = all_tokens
        vocab_size = len(all_tokens)
        print(f"  Vocabulary size: {vocab_size:,} tokens")

        # Initialise embeddings — small random baseline
        emb = np.random.randn(vocab_size, DNA_DIM).astype(np.float32) * 0.02

        # Overwrite with dictionary DNA where we have a match
        print(f"  Loading dictionary DNA vectors...")
        dict_paths = glob.glob(os.path.join(dict_cell_dir, 'meth_english*.json'))
        found = 0
        for path in dict_paths:
            try:
                with open(path) as f:
                    d = json.load(f)
                m = re.match(r'Define:\s+(\w+)', d.get('content', ''))
                if not m:
                    continue
                word = m.group(1).lower()
                if word not in self.word2idx:
                    continue
                dna = d.get('dna', [])
                if len(dna) != DNA_DIM:
                    continue
                vec = np.array(dna, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    emb[self.word2idx[word]] = vec / norm
                    found += 1
            except Exception:
                continue

        initialized = np.sum(np.any(emb != emb[0], axis=1))  # rows that differ from default
        print(f"  Initialised {found:,} dictionary matches → {initialized:,}/{vocab_size:,} unique slots")
        self.embeddings = emb
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'word2idx':  self.word2idx,
            'idx2word':  self.idx2word,
            'embeddings': self.embeddings.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  Vocabulary saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'IAVocabulary':
        with open(path) as f:
            data = json.load(f)
        v = cls()
        v.word2idx  = data['word2idx']
        v.idx2word  = data['idx2word']
        v.embeddings = np.array(data['embeddings'], dtype=np.float32)
        return v
