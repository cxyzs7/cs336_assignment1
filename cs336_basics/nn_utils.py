from collections.abc import Iterable
import torch


def softmax(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """
    Apply the softmax operation on a tensor
    """
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val
    return x.exp() / x.exp().sum(dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy loss.
    """
    max_val = inputs.max(dim=-1, keepdim=True).values
    x = inputs - max_val
    neg_log_prob = -x[torch.arange(inputs.shape[0]), targets] + torch.log(x.exp().sum(dim=-1))
    return neg_log_prob.mean()

    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    """
    params = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum((p.grad*p.grad).sum() for p in params))
    if total_norm > max_l2_norm:
        for p in params:
            p.grad *= max_l2_norm / (total_norm + 1e-6)
