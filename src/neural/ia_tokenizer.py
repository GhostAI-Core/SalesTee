"""
IA Tokenizer
============
GPT-2 BPE via tiktoken — 50,257 tokens covering all English text.
Stateless: no training required, no retraining ever needed.

The vocabulary is fixed and complete. Any word in any future cell
is representable without touching this file or the decoder weights.

Install: pip install tiktoken
"""
import os
import json

try:
    import tiktoken
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class IATokenizer:
    """Thin wrapper around tiktoken GPT-2 BPE."""

    def __init__(self):
        if not _AVAILABLE:
            raise ImportError(
                "tiktoken is required for generation: pip install tiktoken"
            )
        self._enc    = tiktoken.get_encoding("gpt2")
        # Two extra IDs appended beyond the base vocab
        self.bos_id  = self._enc.n_vocab        # 50257
        self.eos_id  = self._enc.n_vocab + 1    # 50258
        self.pad_id  = 50256                    # GPT-2 endoftext — repurposed as PAD
        self.vocab_size = self._enc.n_vocab + 2  # 50259

    def encode(self, text: str) -> list[int]:
        """Text → [BOS, t1, t2, ..., EOS]"""
        return [self.bos_id] + self._enc.encode(text) + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        """Token IDs → text, stripping special tokens."""
        clean = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self._enc.decode(clean)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'type': 'tiktoken_gpt2', 'vocab_size': self.vocab_size}, f)
        print(f"  Tokenizer config saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'IATokenizer':
        return cls()   # stateless — config file is just a marker
