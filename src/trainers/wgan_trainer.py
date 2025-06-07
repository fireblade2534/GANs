import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import autograd

from src.base.base_trainer import BaseTrainer
from src.logging import *
from src.models.wgan import WDiscriminator, WGenerator


class WGanTrainer(BaseTrainer):
    def __init__(self, generator: WGenerator, discriminator: WDiscriminator, generator_optimizer: Optimizer, discriminator_optimizer: Optimizer, generator_scheduler: LRScheduler, discriminator_scheduler: LRScheduler, device: str = "cpu"):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.device = device

    def _on_train_start(self):
        pass

    def _train_generator_iteration(self, epoch: int, zero_grad: bool):
        
        fake_latent_vector = self.generator.generate_latents(self.training_config.batch_size, device=self.device)
        fake_labels = self.generator.generate_labels(self.training_config.batch_size, device=self.device) if self.conditional else None

        fake_images = self.generator(fake_latent_vector, fake_labels)

        generator_loss = -self.discriminator(self.diff_augment.apply_augmentation(fake_images, epoch), fake_labels).mean().view(-1)
        generator_loss.backward()

        if zero_grad:
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)

        return {"Generator Loss": generator_loss.detach().item()}
    
    def _compute_gp(self, real_images: torch.tensor, real_labels: torch.tensor, fake_images: torch.tensor, fake_labels: torch.tensor):
        batch_size = self.training_config.batch_size
        eps = torch.rand(batch_size, 1, 1, 1).to(self.device)
        image_eps = eps.expand_as(real_images)

        image_interpolation = image_eps * real_images + (1 - image_eps) * fake_images
        image_interpolation.requires_grad_(True)

        if self.conditional:
            real_embeddings = self.discriminator.generate_embedding(real_labels)
            fake_embeddings = self.discriminator.generate_embedding(fake_labels)

            embedding_interpolation = (eps * real_embeddings + (1 - eps) * fake_embeddings)
            embedding_interpolation.requires_grad_(True)

            interp_logits = self.discriminator.forward_without_embedding(image_interpolation, embedding_interpolation)
        else:
            interp_logits = self.discriminator(image_interpolation)

        grad_outputs = torch.ones_like(interp_logits)

        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=image_interpolation,#, embedding_interpolation) if self.conditional else image_interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

    def _train_discriminator_iteration(self, real_images: torch.tensor, fake_images: torch.tensor, zero_grad: bool, epoch: int, real_labels: torch.tensor = None, fake_labels: torch.tensor = None):

        # Train the discriminator
        real_prediction = self.discriminator(self.diff_augment.apply_augmentation(real_images.to(self.device), epoch), real_labels)

        fake_prediction = self.discriminator(self.diff_augment.apply_augmentation(fake_images, epoch), fake_labels)

        mean_discriminator_loss = fake_prediction.mean() - real_prediction.mean()

        gradient = self.training_config.gradient_penalty_weight * self._compute_gp(real_images, real_labels, fake_images, fake_labels)

        discriminator_loss = mean_discriminator_loss + gradient

        discriminator_loss.backward()

        if zero_grad:
            self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad(set_to_none=True)

        return {"Discriminator Loss": discriminator_loss.detach().item(), "Discriminator Real Prediction": real_prediction.mean().detach().item(), "Discriminator Fake Prediction": fake_prediction.mean().detach().item(), "Discriminator Gradient": gradient.detach()}