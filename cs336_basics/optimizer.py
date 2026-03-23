import math
from typing import Callable, Optional
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params=params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros(p.data.shape, device=p.device))
                v = state.get('v', torch.zeros(p.data.shape, device=p.device))
                state['m'] = betas[0]*m + (1-betas[0])*grad
                state['v'] = betas[1]*v + (1-betas[1])*grad*grad
                lr_t = lr*(1-betas[1]**t)**0.5/(1-betas[0]**t) 
                p.data -= lr_t*state['m']/(torch.sqrt(state['v'])+eps)
                p.data *= (1-lr*weight_decay)
                state['t'] = t+1
        return loss


def get_lr_cosine_schedule(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.
    
    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.
    """
    if it < warmup_iters:
        return it/warmup_iters*max_lr
    elif it > cosine_cycle_iters:
        return min_lr
    else:
        return min_lr + (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) * (max_lr-min_lr) / 2
