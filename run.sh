#!/bin/bash
PYTHONPATH=. torchrun --nproc_per_node=2 examples/train_mnist.py

