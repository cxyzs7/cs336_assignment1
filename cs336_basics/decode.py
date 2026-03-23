import argparse
import json
import os
import torch
import typing

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import Transformer
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import softmax


def load_vocab_and_merges(vocab_path: str | os.PathLike,
                          merges_path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {int(idx): bytes.fromhex(token) for token, idx in json.load(f).items()}
    with open(merges_path, 'r', encoding='utf-8') as f:
        merges = [tuple(bytes.fromhex(token) for token in line.rstrip('\n').split(' ', 1))
                  for line in f]
    return vocab, merges


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer=None) -> int:
    obj = torch.load(src)
    model.load_state_dict(state_dict=obj['model'])
    if optimizer:
        optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']


def decode(model: torch.nn.Module,
           context_length: int,
           tokenizer: Tokenizer,
           special_tokens: list[str],
           temperature: float,
           max_len: int,
           cumprob_threshold: float,
           prefix: str,
           device: str):
    tokens = tokenizer.encode(text=prefix)
    special_token_list = [tokenizer.encode(s) for s in special_tokens]
    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            input_tokens = tokens[-context_length:]
            output = model(torch.tensor([input_tokens], device=device))
            logits = output[0, -1, :]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = softmax(sorted_logits / temperature)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumprobs - sorted_probs > cumprob_threshold] = 0
            probs = sorted_probs / sorted_probs.sum()
            next_token = sorted_indices[torch.multinomial(probs, 1)].item()
            tokens.append(next_token)
            if [next_token] in special_token_list:
                break
    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chk_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--merges_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--cumprob_threshold', type=float, default=0.9)
    parser.add_argument('input', type=str, default='')
    args = parser.parse_args()

    # load tokenizer
    vocab, merges = load_vocab_and_merges(args.vocab_path, args.merges_path)
    special_tokens = ['<|endoftext|>']
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    print(f'Loaded tokenizer with vocab size {len(vocab)}')

    # load hyperparameters
    with open(args.config) as f:
        hparams = json.load(f)

    # load model checkpoint
    transformer = Transformer(device='cuda', dtype=torch.float32, **hparams)
    iter = load_checkpoint(args.chk_path, transformer)
    print(f'Loaded model with {iter} iterations')
    
    output = decode(model=transformer,
                    context_length=hparams['context_length'],
                    tokenizer=tokenizer,
                    special_tokens=special_tokens,
                    temperature=args.temperature,
                    max_len=args.max_len,
                    cumprob_threshold=args.cumprob_threshold,
                    prefix=args.input,
                    device='cuda')
    print(output)

if __name__ == '__main__':
    main()
