"""
MiniDistTrainer - Benchmark: single process vs. DDP
Author: matcha-n
Date: 2025-08-29
"""

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms

from minidist.ddp import MiniDDP
from minidist.engine import Engine
from minidist.utils import set_seed


def get_dataloader(rank=0, world_size=1):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    if world_size > 1:
        sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    return DataLoader(dataset, batch_size=64, sampler=sampler)


def build_model(device):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28*28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).to(device)


def run_single():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    dataloader = get_dataloader()
    model = build_model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    engine = Engine(model, optimizer, dataloader, device)

    start = time.time()
    ddp_model = MiniDDP(model)  # works as identity here
    engine.train(epochs=1, ddp_model=ddp_model)
    end = time.time()

    print(f"[Single process] Time: {end - start:.2f}s")


def run_ddp():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42 + rank)

    dataloader = get_dataloader(rank, world_size)
    model = build_model(device)
    ddp_model = MiniDDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    engine = Engine(model, optimizer, dataloader, device)

    start = time.time()
    engine.train(epochs=1, ddp_model=ddp_model)
    end = time.time()

    if rank == 0:
        print(f"[DDP {world_size} processes] Time: {end - start:.2f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "ddp"], default="single")
    parser.add_argument("--nproc", type=int, default=2, help="num processes for DDP")
    args = parser.parse_args()

    if args.mode == "single":
        run_single()
    else:
        # run with torchrun
        run_ddp()
