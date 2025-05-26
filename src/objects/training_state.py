from dataclasses import dataclass
from typing import Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.base.base_gan import BaseDiscriminator, BaseGenerator

@dataclass
class TrainState:
    epoch: int
    generator: Dict
    discriminator: Dict
    generator_optimizer: Optimizer
    discriminator_optimizer: Optimizer
    generator_scheduler: LRScheduler
    discriminator_scheduler: LRScheduler