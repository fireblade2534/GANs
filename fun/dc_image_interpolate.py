import sys
from pathlib import Path

# Debug the paths
script_path = Path(__file__).absolute()
project_root = script_path.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))

from src.models.dcgan import DCGenerator
from src.base.base_trainer import BaseTrainer
import torch
import matplotlib.pyplot as plt
import random

device = "cpu"
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    device = "cuda"

checkpoint = BaseTrainer.load_checkpoint("training_runs/mnist_1/checkpoints/mnist_32x32_1_30_model.pt")

generator = DCGenerator(latent_dimension=128, image_shape=(3,32,32), used_layers=3, total_layers=3)

generator.load_model_state(checkpoint.generator)

generator.to(device)

generator.eval()
for i in range(1):
    test_latents = generator.generate_latents(2, device=device)

    latent_start = test_latents[0]  # Shape: (64, 1, 1)
    latent_end = test_latents[1]    # Shape: (64, 1, 1)

    num_intermediate_points = 20

    interpolated_latents_list = []

    for i in range(0, num_intermediate_points):
        alpha = i / (num_intermediate_points - 1)  # Ensure float division

        # Linear interpolation formula: (1 - alpha) * start + alpha * end
        interpolated_vector = ((1.0 - alpha) * test_latents[0])  + (alpha * test_latents[1])
        interpolated_latents_list.append(interpolated_vector)

    generated_intermediate_latents = torch.stack(interpolated_latents_list, dim=0).to(device)

    with torch.no_grad():
        generated = generator(generated_intermediate_latents.detach()).cpu().permute(0, 2, 3, 1)

        generated = (generated + 1) / 2

        print(generated.shape)

    _, axes = plt.subplots(nrows=1, ncols=num_intermediate_points, figsize=(8, 2))
    plt.suptitle(f"Interpolation")

    for index in range(num_intermediate_points):
        axes[index].imshow(generated[index])
        axes[index].axis("off")

    plt.savefig(f"sample_.png")#{random.randint(0,10000)}.png")
    plt.close()