from dataclasses import asdict
import torch
from src.logging import *
from src.objects.training_config import AugmentationConfig
import torch.nn.functional as Functional

class DiffAugment:

    def __init__(self, config: AugmentationConfig, image_shape: torch.tensor, device: str):
        augmentation_map = {"cutout": self.agumentation_cutout, "translation": self.agumentation_translation, "noise": self.augment_noise}

        self.image_shape = image_shape
        self.device = device
        self.augmentations = []
        self.config = config

        for augmentation, state in asdict(config).items():
            if state["enabled"]:
                if augmentation in augmentation_map:
                    self.augmentations.append(augmentation_map[augmentation])
                else:
                    logger.Warning(f"No function candidate found for augmentation: {augmentation}")

        self.x_indexes = torch.arange(image_shape[2], device=device).view(1, 1, 1, -1)
        self.y_indexes = torch.arange(image_shape[1], device=device).view(1, 1, -1, 1)
        self.channel_indexes = torch.arange(image_shape[0], device=device).view(1, -1, 1, 1)

    @torch.compile
    def apply_agumentation(self, image_tensor: torch.tensor, epoch: int):

        for augmentation in self.augmentations:
            image_tensor = augmentation(image_tensor, epoch)
        return image_tensor

    @torch.compile
    def agumentation_cutout(self, image_tensor: torch.tensor, epoch: int):
        ratio = self.config.cutout.ratio

        cutout_size = (int((self.image_shape[1] * ratio) + 0.5), int((self.image_shape[2] * ratio) + 0.5))

        mask_tensor = torch.ones_like(image_tensor)

        y_start_index = torch.randint(0, self.image_shape[2] - cutout_size[1], size=(image_tensor.size(0),1,1,1), device=self.device)
        x_start_index = torch.randint(0, self.image_shape[1] - cutout_size[0], size=(image_tensor.size(0),1,1,1), device=self.device)

        y_end_index = y_start_index + cutout_size[1]
        x_end_index = x_start_index + cutout_size[0]

        channel_index = torch.randint(0, 3, size=(image_tensor.size(0),1,1,1), device=self.device)

        mask_broadcast = (self.channel_indexes == channel_index) & (self.x_indexes >= x_start_index) & (self.x_indexes < x_end_index) & (self.y_indexes >= y_start_index) & (self.y_indexes < y_end_index)

        mask_tensor[mask_broadcast] = 0

        return image_tensor * mask_tensor

    @torch.compile
    def agumentation_translation(self, image_tensor: torch.tensor, epoch: int):
        ratio = self.config.translation.ratio

        max_transformation_size = min(int((image_tensor.size(2) * ratio) + 0.5), int((image_tensor.size(3) * ratio) + 0.5))

        transformation = torch.randint(-max_transformation_size, max_transformation_size + 1, size=(image_tensor.size(0), 2), device=self.device)

        theta = torch.zeros(image_tensor.size(0), 2, 3, device=self.device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 0, 2] = -(transformation[:,0].float() / self.image_shape[2]) * 2
        theta[:, 1, 2] = -(transformation[:,1].float() / self.image_shape[1]) * 2

        affine_transformation_grid = Functional.affine_grid(theta, image_tensor.size(), align_corners=False)

        output_image_tensor = Functional.grid_sample(image_tensor + 1, affine_transformation_grid, mode='bilinear', 
                           padding_mode='zeros', align_corners=False)
    
        return output_image_tensor - 1

    @torch.compile
    def augment_noise(self, image_tensor: torch.tensor, epoch: int):
        starting_noise = self.config.noise.starting_noise
        ending_noise = self.config.noise.ending_noise
        ending_epoch = self.config.noise.ending_epoch

        epoch_percent = min(epoch / ending_epoch, 1)
        noise_magnitude = (starting_noise * (1 - epoch_percent)) + (ending_noise * epoch_percent)

        noise = torch.randn_like(image_tensor, device=self.device) * noise_magnitude

        return torch.clamp(image_tensor + noise, -1, 1)