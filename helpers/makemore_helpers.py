from __future__ import annotations
import torch


def make_char2idx_map(words: list[str], delimiter='.') -> dict[str,int]:
    """
    Make a mapping of characters to indices.
    """
    chars = [delimiter] + sorted(list(set(''.join(words))))
    char2idx_map = {c: i for i, c in enumerate(chars)}
    return char2idx_map

def build_dataset(words: list[str], block_size: int, char2idx_map : dict[str,int], delimiter = '.') -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dataset from a list of words.

    Args:
        words (list[str]): List of words.
        block_size (int): Size of the blocks to split the words into.
        char2idx_map (dict[str,int]): Mapping of characters to indices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of tensors containing the input and target blocks.
    """
    xs, ys = [], []
    for word in words:
        context = [char2idx_map[delimiter]] * block_size
        
        for _chr in word+delimiter:
            ix = char2idx_map[_chr]
            xs.append(context)
            ys.append(ix)

            context = context[1:] + [ix]

    return torch.tensor(xs), torch.tensor(ys)


  