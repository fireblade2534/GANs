from matplotlib.pyplot import cla
from src.trainers.dcgan_trainer import DCGanTrainer
from src.models.dcgan import DCGenerator, DCDiscriminator

from src.objects.training_config import AugmentationConfig, AugmentationCutoutConfig, AugmentationNoiseConfig, AugmentationTranslationConfig, TrainingConfig
import torch
import torchvision
import torchvision.transforms as transforms

device = "cpu"
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True 
    device = "cuda"

img_size = 32
img_channels = 3
img_shape = (img_channels, img_size, img_size)

image_transform = transforms.Compose(
    [  # transforms.Resize(img_size), # Resize is only for PIL Image. Not for numpy array
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # ToTensor() : np.array (H, W, C) -> tensor (C, H, W)
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
    ]
)
trainset = torchvision.datasets.FashionMNIST(root="~/Datasets/Images", train=True, download=True, transform=image_transform)

generator = DCGenerator(latent_dimension=64, used_layers=3, total_layers=5, image_shape=img_shape, conv_dimension=96, num_labels=10)
discriminator = DCDiscriminator(used_layers=3, total_layers=5, image_shape=img_shape, conv_dimension=96, num_labels=10)

training_config = TrainingConfig(
    generator_learning_rate=0.00009,
    discriminator_learning_rate=0.00009,
    b1=0.5,
    b2=0.99,
    batch_size=64,
    epochs=200,
    sample_epochs=2,
    save_epochs=2,
    discriminator_repeats=3,
    gradient_penalty_weight=10,
    gradient_accumulation_steps=1,
    stablization_epochs=2,
    num_data_workers=16,
    num_labels=10,
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

trainer = DCGanTrainer(generator, discriminator, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator_scheduler=generator_scheduler, discriminator_scheduler=discriminator_scheduler, device=device)

trainer.train("fasion_mnist_32x32_1", "training_runs/fasion_mnist_1", training_config, trainset, override_resume_options=False)#, resume_path="training_runs/stanford_dogs/checkpoints/stanford_dogs_64x64_5_124_model.pt")