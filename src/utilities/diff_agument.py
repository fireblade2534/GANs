from dataclasses import asdict
import torch
from src.logging import *
from src.objects.training_config import AugmentationConfig

class DiffAugment:

    def __init__(self, config: AugmentationConfig, image_shape: torch.tensor, device: str):
        augmentation_map = {"cutout": self.agumentation_cutout}

        self.image_shape = image_shape
        self.device = device
        self.augmentations = []

        for augmentation, state in asdict(config).items():
            if state:
                if augmentation in augmentation_map:
                    self.augmentations.append(augmentation_map[augmentation])
                else:
                    logger.Warning(f"No function candidate found for augmentation: {augmentation}")

        self.x_indexes = torch.arange(image_shape[2], device=device).view(1, 1, 1, -1)
        self.y_indexes = torch.arange(image_shape[1], device=device).view(1, 1, -1, 1)
        self.channel_indexes = torch.arange(image_shape[0], device=device).view(1, -1, 1, 1)

    @torch.compile
    def apply_agumentation(self, image_tensor: torch.tensor):

        for augmentation in self.augmentations:
            image_tensor = augmentation(image_tensor)
        return image_tensor

    @torch.compile
    def agumentation_cutout(self, image_tensor: torch.tensor, ratio: float = 0.5):
        cutout_size = (int((image_tensor.size(3) * ratio) + 0.5), int((image_tensor.size(2) * ratio) + 0.5))

        mask_tensor = torch.ones_like(image_tensor)

        y_start_index = torch.randint(0, image_tensor.size(3), size=(image_tensor.size(0),1,1,1), device=self.device)
        x_start_index = torch.randint(0, image_tensor.size(2), size=(image_tensor.size(0),1,1,1), device=self.device)

        y_end_index = y_start_index + cutout_size[1]
        x_end_index = x_start_index + cutout_size[0]

        channel_index = torch.randint(0, 3, size=(image_tensor.size(0),1,1,1), device=self.device)

        mask_broadcast = (self.channel_indexes == channel_index) & (self.x_indexes >= x_start_index) & (self.x_indexes < x_end_index) & (self.y_indexes >= y_start_index) & (self.y_indexes < y_end_index)

        mask_tensor[mask_broadcast] = 0

        return image_tensor * mask_tensor
