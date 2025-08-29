"""
MiniDistTrainer - A minimal distributed training framework

Author: matcha-n
Email: matcha.nuoran@gmail.com
Created: 2025-08-29
"""

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

class MiniDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, use_checkpoint: bool = False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, *inputs, **kwargs):
        if self.use_checkpoint:
            return checkpoint(self.module, *inputs, **kwargs)
        return self.module(*inputs, **kwargs)

    def backward(self, loss):
        loss.backward()
        self._average_gradients()

    def _average_gradients(self):
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
