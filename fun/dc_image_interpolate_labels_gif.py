import sys
from pathlib import Path
import imageio.v3 as iio
import imageio

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

checkpoint = BaseTrainer.load_checkpoint("training_runs/mnist_5/checkpoints/mnist_32x32_5_50_model.pt")

generator = DCGenerator(latent_dimension=16, image_shape=(3,32,32), used_layers=3, total_layers=3, conv_dimension=64, num_labels=10)

generator.load_model_state(checkpoint.generator)

generator.to(device)

generator.eval()
for i in range(5):
    test_latents = generator.generate_latents(2, device=device)

    test_labels = generator.generate_labels(2, device=device)
    test_embeddings = generator.generate_embedding(test_labels)

    num_intermediate_points = 50

    interpolated_latents_list = []
    interploated_embeddings_list = []

    for i in range(0, num_intermediate_points):
        alpha = i / (num_intermediate_points - 1)  # Ensure float division

        # Linear interpolation formula: (1 - alpha) * start + alpha * end
        interpolated_vector = ((1.0 - alpha) * test_latents[0])  + (alpha * test_latents[1])
        interpolated_latents_list.append(interpolated_vector)

        interpolated_embedding = ((1.0 - alpha) * test_embeddings[0])  + (alpha * test_embeddings[1])
        interploated_embeddings_list.append(interpolated_embedding)

    generated_intermediate_latents = torch.stack(interpolated_latents_list, dim=0).to(device)
    generated_intermediate_embeddings = torch.stack(interploated_embeddings_list, dim=0).to(device)

    with torch.no_grad():
        generated = generator.forward_without_embedding(generated_intermediate_latents.detach(), generated_intermediate_embeddings.detach()).cpu()
        print(generated.shape)

    last_frame = generated[-1:].repeat(20, 1, 1, 1)
    generated = torch.cat([generated, last_frame, torch.flip(generated, dims=(0,))], dim=0).repeat(4, 1, 1, 1)

    generated = generated.permute(0, 2, 3, 1)
    fix_generated = ((generated + 1) * 128).to(torch.uint8)
    #exit()
    imageio.mimwrite(f'sample_{random.randint(0,10000)}.gif', fix_generated.numpy())