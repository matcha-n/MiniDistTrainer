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

## 3. Quick Start
Run the MNIST training example with 2 processes: 
```bash
cd MiniDistTrainer   # go to project root
PYTHONPATH=. torchrun --nproc_per_node=2 examples/train_mnist.py
```
Example output (loss printed by rank 0):
```
[Epoch 0] Step 0   | Loss: 2.2878
[Epoch 0] Step 100 | Loss: 2.1875
[Epoch 0] Step 200 | Loss: 1.9878
[Epoch 0] Step 300 | Loss: 1.7580
[Epoch 0] Step 400 | Loss: 1.4879
[Epoch 1] Step 0   | Loss: 1.3103
[Epoch 1] Step 100 | Loss: 1.0309
[Epoch 1] Step 200 | Loss: 0.9256
[Epoch 1] Step 300 | Loss: 0.7869
[Epoch 1] Step 400 | Loss: 0.7236
```
The loss decreases steadily and shows that the training and gradient synchronization are working correctly.

## 4. How It Works
1. Each process maintains its own copy of the model.
2. Forward pass is computed independently.
3. After backward pass, gradients are averaged across all processes via AllReduce.
4. It makes that the training is equivalent to single-GPU training with a larger batch size.
A core snippet (from `MiniDDP`):
```python
for param in self.module.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= dist.get_world_size()
```


## 5. Project Structure
```
MiniDistTrainer/
│── README.md
│── requirements.txt
│── run.sh
│── minidist/
│   │── __init__.py
│   │── ddp.py         # minimal DDP implementation
│   │── engine.py      # training engine
│   │── utils.py       # utility functions
│── examples/
│   │── train_mnist.py # MNIST training example
│── tests/
│   │── test_ddp.py    # unit tests

```

## 6. Goal
The project is not a replacement for PyTorch DDP. Its goal is to help newcomers understand distributed training internals while serving as a neat showcase for AI Infrastructure engineering skills.

## 7. Author
+ Name: matcha-n
+ Email: matcha.nuoran@gmail.com
+ Date Created: 2025-08-29

Keep eager to transition into **llm backend development**. And hopefully, one day, may have the opportunity to also work on **AI&ML algorithm**.
