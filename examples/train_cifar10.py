"""
MiniDistTrainer - Example: Train CIFAR-10 with MiniDDP
Author: matcha-n
Date: 2025-08-29
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms

from minidist.ddp import MiniDDP
from minidist.engine import Engine
from minidist.utils import set_seed


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42 + rank)

    # CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # A Simple CNN model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1), torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(64*8*8, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).to(device)

    ddp_model = MiniDDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    engine = Engine(model, optimizer, dataloader, device)
    engine.train(epochs=2, ddp_model=ddp_model)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
