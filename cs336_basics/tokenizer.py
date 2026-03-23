from collections.abc import Iterable
import json
import regex as re
from typing import Iterator

PAT= r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.vocab_index = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.merge_idx = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges 
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_json = json.load(f)
            vocab = {int(k): bytes(v) for k, v in vocab_json.items}
        
        with open(merges_filepath, encoding="utf-8") as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        encoded_tokens: list[int] = []
        # Handle special tokens
        # Sort by length descending so longer special tokens match first
        if self.special_tokens:
            special_pat = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
            chunks = special_pat.split(text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if chunk in self.special_tokens:
                encoded_tokens.append(self.vocab_index[chunk.encode("utf-8")])
            else:
                # Pre-tokenize
                for m in COMPILED_PAT.finditer(chunk):
                    merged_token = tuple([bytes([b]) for b in m.group().encode("utf-8")])
                    # Apply the merges
                    voc_max_idx = len(self.vocab)
                    while True:
                        voc_idx = voc_max_idx
                        token_idx = -1
                        # Find the lowest merge index
                        for i in range(len(merged_token)-1):
                            idx = self.merge_idx.get((merged_token[i], merged_token[i+1]), voc_max_idx)
                            if idx < voc_idx:
                                voc_idx = idx
                                token_idx = i
                        # If no available merge
                        if voc_idx == voc_max_idx:
                            break
                        # Merge
                        merged_token = (
                            merged_token[:token_idx] 
                            + tuple([merged_token[token_idx] + merged_token[token_idx+1]])
                            + merged_token[token_idx+2:]
                        )
                        # If the entire pre-token has been merged
                        if (len(merged_token) == 1):
                            break
                    
                    # Find the id
                    for token in merged_token:
                        encoded_tokens.append(self.vocab_index[token])

        return encoded_tokens
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        tokens = bytes()
        for id in ids:
            token = self.vocab.get(id, bytes(b""))
            tokens += token
        return tokens.decode("utf-8", errors="replace")
        