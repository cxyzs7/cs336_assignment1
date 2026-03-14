import os
import regex as re
from collections import defaultdict, Counter
from typing import BinaryIO

PAT= r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Init vocab and merges
    vocab:dict[int, bytes] = (
        {i: t.encode("utf-8") for i, t in enumerate(special_tokens)} 
        | {len(special_tokens) + i: bytes([i]) for i in range(256)}
    )
    merges:list[tuple[bytes, bytes]] = []
    
    with open(input_path, "rb") as f:
        # Pre-tokenization
        pre_token_counts_total = Counter()
        # Define parallel processing for pre-tokenization
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        split_pat = re.compile("|".join(re.escape(t) for t in special_tokens)) if special_tokens else None
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # Removing special tokens before pre-tokenization, "|"" may occur in the special tokens
            chunk_splits = split_pat.split(chunk) if split_pat else [chunk]
            
            # Run pre-tokenization on each chunk and store the counts for each pre-token
            pre_token_counts = Counter(
                m.group()
                for split in chunk_splits
                for m in COMPILED_PAT.finditer(split)
            )
            
            # Accumulate counts
            pre_token_counts_total += pre_token_counts

        # Compute BPE merges
        pre_tokens = {}
        pair_counts = Counter()
        pair_to_pretokens = defaultdict(set)
        # Index the initial pairs
        for pre_token, freq in pre_token_counts_total.items():
            pre_token_tuple = tuple(bytes([b]) for b in pre_token.encode('utf-8'))
            pre_tokens[pre_token_tuple] = freq
            for a, b in zip(pre_token_tuple[:-1], pre_token_tuple[1:]):
                pair_counts[(a, b)] += freq
                pair_to_pretokens[(a, b)].add(pre_token_tuple)

        # Merge until meeting vocab size
        for i in range(vocab_size - len(special_tokens) - 256):
            # Find most frequent, use lexicographically order to break tie
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = merged_token
            # Incrementally update counts
            for pre_token in pair_to_pretokens[best_pair]:
                freq = pre_tokens[pre_token]
                new_pre_token = []
                j = 0
                while j < len(pre_token):
                    if j < len(pre_token) - 1 and (pre_token[j], pre_token[j+1]) == best_pair:
                        # Update counts: remove left neighbor pair, add new left neighbor pair
                        if new_pre_token:
                            pair_counts[(new_pre_token[-1], best_pair[0])] -= freq
                            pair_counts[(new_pre_token[-1], merged_token)] += freq
                        # Remove right neighbor pair, add new right neighbor pair
                        if j + 2 < len(pre_token):
                            pair_counts[(best_pair[1], pre_token[j+2])] -= freq
                            pair_counts[(merged_token, pre_token[j+2])] += freq
                        new_pre_token.append(merged_token)
                        j += 2
                    else:
                        new_pre_token.append(pre_token[j])
                        j += 1
                new_pre_token = tuple(new_pre_token)
                # Update pre_tokens
                del pre_tokens[pre_token]
                pre_tokens[new_pre_token] = freq
                # Update pair_to_pretokens
                for a, b in zip(pre_token[:-1], pre_token[1:]):
                    if (a, b) != best_pair:
                        pair_to_pretokens[(a, b)].discard(pre_token)
                for a, b in zip(new_pre_token[:-1], new_pre_token[1:]):
                    pair_to_pretokens[(a, b)].add(new_pre_token)

            del pair_counts[best_pair]
            
        return vocab, merges