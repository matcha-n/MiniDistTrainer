#!/bin/bash
PYTHONPATH=. torchrun --nproc_per_node=2 examples/train_mnist.py
PYTHONPATH=. torchrun --nproc_per_node=2 examples/train_cifar10.py


