from src.trainers.wgan_trainer import WGanTrainer
from src.models.wgan import WGenerator, WDiscriminator

from src.objects.training_config import AugmentationConfig, AugmentationCutoutConfig, AugmentationNoiseConfig, AugmentationTranslationConfig, TrainingConfig
import torch
import torchvision
import torchvision.transforms as transforms

device = "cpu"
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True 
    device = "cuda"

img_size = 64
img_channels = 3
img_shape = (img_channels, img_size, img_size)

image_transform = transforms.Compose(
    [  # transforms.Resize(img_size), # Resize is only for PIL Image. Not for numpy array
        transforms.ToTensor(),  # ToTensor() : np.array (H, W, C) -> tensor (C, H, W)
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
    ]
)
temp = torchvision.datasets.CelebA

temp._check_integrity = lambda x : True

trainset = temp(root="~/Datasets/Images", download=False, target_type="attr", transform=image_transform)

generator = WGenerator(latent_dimension=100, used_layers=4, total_layers=4, image_shape=img_shape, conv_dimension=32, num_labels=40, one_hot_labels=False)
discriminator = WDiscriminator(used_layers=4, total_layers=4, image_shape=img_shape, conv_dimension=32, num_labels=40, one_hot_labels=False)

training_config = TrainingConfig(
    generator_learning_rate=0.00005,
    discriminator_learning_rate=0.00005,
    b1=0.0,
    b2=0.99,
    batch_size=64,
    epochs=50,
    sample_epochs=2,
    save_epochs=2,
    discriminator_repeats=5,
    gradient_penalty_weight=10,
    gradient_accumulation_steps=1,
    stablization_epochs=2,
    num_data_workers=16,
    num_labels=40,
    augmentation_config=AugmentationConfig(
        translation=AugmentationTranslationConfig(enabled=True),
        cutout=AugmentationCutoutConfig(enabled=True),
        noise=AugmentationNoiseConfig(enabled=True)
    )
)


generator_optimizer = torch.optim.Adam(generator.parameters(), lr=training_config.generator_learning_rate, betas=(training_config.b1, training_config.b2))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=training_config.discriminator_learning_rate, betas=(training_config.b1, training_config.b2))

generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=training_config.epochs, eta_min=training_config.generator_learning_rate*0.001)
discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=training_config.epochs, eta_min=training_config.discriminator_learning_rate*0.001)

trainer = WGanTrainer(generator, discriminator, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator_scheduler=generator_scheduler, discriminator_scheduler=discriminator_scheduler, device=device)

trainer.train("celeb_a_64x64_1", "training_runs/celeb_a_1", training_config, trainset, override_resume_options=False)#, resume_path="training_runs/celeb_a_1/checkpoints/celeb_a_64x64_1_6_model.pt")