from dataclasses import dataclass, field

@dataclass
class AugmentationConfig:
    color: bool = False
    translation: bool = False
    cutout: bool = False

@dataclass
class TrainingConfig:
    seed: int = 42
    generator_learning_rate: float = 0.0002
    discriminator_learning_rate: float = 0.0004
    batch_size: int = 64
    epochs: int = 20
    b1: float = 0.5
    b2: float = 0.999
    sample_grid_size: int = 6
    sample_epochs: int = 2
    save_epochs: int = 2
    num_data_workers: int = 8
    stablization_epochs: int = 6
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    gradient_penalty_weight: int = 10
    discriminator_repeats: int = 3
    gradient_accumulation_steps: int = 2