"""
MiniDistTrainer - Example: Train MNIST with MiniDDP

Author: matcha-n
Email: matcha.nuoran@gmail.com
Created: 2025-08-29
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms

from minidist.ddp import MiniDDP
from minidist.engine import Engine
from minidist.utils import set_seed


def main():
    # initialize distributed backend
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42 + rank)

    # dataset & sampler
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # model & optimizer
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28*28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).to(device)

    ddp_model = MiniDDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # train
    engine = Engine(model, optimizer, dataloader, device)
    engine.train(epochs=2, ddp_model=ddp_model)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
