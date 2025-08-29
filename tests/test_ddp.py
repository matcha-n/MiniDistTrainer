"""
MiniDistTrainer - Tests for MiniDDP

Author: matcha-n
Email: matcha.nuoran@gmail.com
Created: 2025-08-29
"""

import torch
import torch.distributed as dist
from minidist.ddp import MiniDDP


def dist_setup():
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


def test_ddp_backward():
    dist_setup()
    model = torch.nn.Linear(10, 1)
    ddp_model = MiniDDP(model)

    x = torch.randn(4, 10)
    y = torch.randn(4, 1)

    out = ddp_model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    ddp_model.backward(loss)

    for p in model.parameters():
        assert p.grad is not None
