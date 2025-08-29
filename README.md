# MiniDistTrainer 🚀

A lightweight distributed training framework built on top of PyTorch.  

The project demonstrates the core principles of **Distributed Data Parallel (DDP)** training in under 500 lines of code.  It provides a simple yet educational implementation of gradient synchronization via AllReduce and allows multiple processes to train models collaboratively.

## 1. Features
1. Minimal DDP implementation in <500 lines.
2. Gradient averaging with `torch.distributed.all_reduce`.
3. Train engine abstraction for clean loops.
4. Example: distributed MNIST training with `torchrun`.
5. Easy to extend for learning and experimentation.

## 2. Installation
```bash
git clone https://github.com/matcha-n/MiniDistTrainer.git
cd MiniDistTrainer
pip install -r requirements.txt
```
Requirements:
+ Python >= 3.8
+ PyTorch >= 1.12
+ torchvision (for MNIST example)
