"""
MiniDistTrainer - Training Engine

Author: matcha-n
Email: matcha.nuoran@gmail.com
Created: 2025-08-29
"""

import torch
import torch.distributed as dist

class Engine:
    """
    Simple training engine that runs the training loop.
    Works with MiniDDP for distributed gradient synchronization.
    """
    def __init__(self, model, optimizer, dataloader, device):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device

    def train(self, epochs, ddp_model):
        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x, y = x.to(self.device), y.to(self.device)

                out = ddp_model(x)
                loss = torch.nn.functional.cross_entropy(out, y)

                self.optimizer.zero_grad()
                ddp_model.backward(loss)
                self.optimizer.step()

                if dist.get_rank() == 0 and batch_idx % 100 == 0:
                    print(f"[Epoch {epoch}] Step {batch_idx} | Loss: {loss.item():.4f}")
