from src.base.base_trainer import BaseTrainer
from src.models.dcgan import DCGenerator, DCDiscriminator
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.objects.training_config import TrainingConfig
import os
import torch
from src.logging import *

class DCGanTrainer(BaseTrainer):
    def __init__(self, generator: DCGenerator, discriminator: DCDiscriminator, generator_optimizer: Optimizer, discriminator_optimizer: Optimizer, generator_scheduler: LRScheduler, discriminator_scheduler: LRScheduler, device: str = "cpu"):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.device = device
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()

    def _on_train_start(self):
        self.valid_base = torch.full((self.training_config.batch_size, 1), 0.9, device=self.device, requires_grad=False)
        self.fake_base = torch.full((self.training_config.batch_size, 1), 0.1, device=self.device, requires_grad=False)

    def _train_generator_iteration(self, zero_grad: bool):
        valid = torch.clamp(self.valid_base + 0.1 * torch.randn_like(self.valid_base, device=self.device), 0, 1)

        fake_latent_vector = self.generator.generate_latents(self.training_config.batch_size, device=self.device)
        fake_labels = self.generator.generate_labels(self.training_config.batch_size, device=self.device)

        fake_images = self.generator(fake_latent_vector, fake_labels)

        generator_loss = self.adversarial_loss(self.discriminator(self.diff_augment.apply_agumentation(fake_images), fake_labels), valid)
        generator_loss.backward()

        if zero_grad:
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)

        return {"Generator Loss": generator_loss.item()}

    def _train_discriminator_iteration(self, real_images: torch.tensor, fake_images: torch.tensor, zero_grad: bool, real_labels: torch.tensor = None, fake_labels: torch.tensor = None):
        valid = torch.clamp(self.valid_base + 0.1 * torch.randn_like(self.valid_base, device=self.device), 0, 1)
        fake = torch.clamp(self.fake_base + 0.1 * torch.randn_like(self.fake_base, device=self.device), 0, 1)
        
        # Train the discriminator
        real_prediction = self.discriminator(self.diff_augment.apply_agumentation(real_images.to(self.device)), real_labels)
        real_loss = self.adversarial_loss(real_prediction, valid)

        fake_prediction = self.discriminator(self.diff_augment.apply_agumentation(fake_images), fake_labels)
        fake_loss = self.adversarial_loss(fake_prediction, fake)

        discriminator_loss = (real_loss + fake_loss) / 2

        discriminator_loss.backward()

        if zero_grad:
            self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad(set_to_none=True)

        return {"Discriminator Loss": discriminator_loss.item(), "Discriminator Real Loss": real_loss.item(), "Discriminator Fake Loss": fake_loss.item(), "Discriminator Real Prediction": real_prediction.mean().item(), "Discriminator Fake Prediction": fake_prediction.mean().item()}