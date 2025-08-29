"""
MiniDistTrainer - A minimal distributed training framework

Author: matcha-n
Email: matcha.nuoran@gmail.com
Created: 2025-08-29
"""

# make submodules available
from .ddp import MiniDDP
from .engine import Engine