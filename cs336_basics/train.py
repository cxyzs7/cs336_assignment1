import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import typing

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe
from cs336_basics.transformer import Transformer
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


def get_batch(dataset: np.array, batch_size: int, context_length: int, device: str):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    """
    # Sample random starting positions
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.tensor(np.stack([dataset[s:s+context_length] for s in starts]), device=device)
    y = torch.tensor(np.stack([dataset[s+1:s+context_length+1] for s in starts]), device=device)
    return x, y


def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                          vocab_out: str | os.PathLike,
                          merges_out: str | os.PathLike):
    with open(vocab_out, 'w', encoding='utf-8') as f:
        json.dump({token.hex(): idx for idx, token in vocab.items()}, f)
    with open(merges_out, 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write(a.hex() + ' ' + b.hex() + '\n')


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--vocab_input_path', type=str, default=None)
    parser.add_argument('--save_vocab', action='store_true', default=True)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--rope_theta', type=float, default=10000)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=40000)
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    vocab_input_path = args.vocab_input_path if args.vocab_input_path is not None else input_path
    save_vocab = args.save_vocab
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    context_length = args.context_length
    d_model = args.d_model
    d_ff = args.d_ff
    rope_theta = args.rope_theta
    num_layers = args.num_layers
    num_heads = args.num_heads
    num_steps = args.num_steps
    
    dataset_name = Path(input_path).stem

    #
    torch.manual_seed(42)

    # get tokenizer
    print('Training tokenizer')
    special_tokens = ['<|endoftext|>']
    vocab, merges = train_bpe(input_path=vocab_input_path,
                              vocab_size=vocab_size,
                              special_tokens=special_tokens)
    if save_vocab:
        vocab_output_path = Path(output_dir, f'{dataset_name}_vocab_{len(vocab)}.json')
        merges_output_path = Path(output_dir, f'{dataset_name}_merges_{len(vocab)}.txt')
        save_vocab_and_merges(vocab, merges, vocab_output_path, merges_output_path)
        print('saved vocab and merges to {} and {}'.format(str(vocab_output_path.absolute), str(merges_output_path.absolute)))
    tokenizer = Tokenizer(vocab=vocab,
                          merges=merges,
                          special_tokens=special_tokens)
    print('Tokenizer trained')
    
    # get dataset
    # TODO: Memory-efficient loading of training and validation large datasets with np.memmap.
    print('Loading dataset')
    with open(input_path) as f:
        texts = f.read()
        dataset = tokenizer.encode(text=texts)
    print('Dataset loaded')

    # save hyperparameters
    hparams = {
        'vocab_size': len(vocab),
        'context_length': context_length,
        'd_model': d_model,
        'd_ff': d_ff,
        'rope_theta': rope_theta,
        'num_layers': num_layers,
        'num_heads': num_heads,
    }
    with open(Path(output_dir, f'{dataset_name}_config.json'), 'w') as f:
        json.dump(hparams, f, indent=2)

    # define model and optimizer
    transformer = Transformer(vocab_size=len(vocab),
                              context_length=context_length,
                              d_model=d_model,
                              num_layers=num_layers,
                              num_heads=num_heads,
                              d_ff=d_ff,
                              rope_theta=rope_theta,
                              device='cuda',
                              dtype=torch.float32)
    opt = AdamW(
        transformer.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # train
    # TODO: learning schedule, gradient clipping
    for it in range(num_steps):
        x, y = get_batch(dataset=dataset,
                         batch_size=batch_size,
                         context_length=context_length,
                         device='cuda')
        opt.zero_grad()
        y_hat = transformer(x)
        loss = cross_entropy(y_hat.view(-1, y_hat.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        print(f'step:{(it+1)} loss:{loss.item():.4f}')
        # save every 500 steps
        # TODO: more detailed logging
        if (it+1) % 500 == 0:
            chkp_output_path = Path(output_dir, f'{dataset_name}_chkp_{(it+1)}')
            save_checkpoint(transformer, opt, it, chkp_output_path)


if __name__ == '__main__':
    main()
