from src.base.base_gan import BaseDiscriminator, BaseGenerator
from src.utilities.weights import init_weight
import torch.nn as nn
import torch
from typing import OrderedDict
import math

class WGenerator(BaseGenerator):
    def __init__(self, latent_dimension: int, image_shape: tuple, used_layers: int, total_layers: int, conv_dimension: int = 48, num_labels: int = 0):
        super(WGenerator, self).__init__()

        self.latent_dimension = latent_dimension
        self.image_shape = image_shape
        self.used_layers = used_layers
        self.total_layers = total_layers
        self.conv_dimension = conv_dimension

        self.num_labels = num_labels
        self.conditional = num_labels > 0

        input_dimension = self.latent_dimension
        if self.conditional:
            self.label_embedding = nn.Sequential(nn.Linear(self.num_labels, latent_dimension * 2), nn.Linear(latent_dimension * 2, latent_dimension))
            
            input_dimension *= 2

        multiplier = 2 ** total_layers

        temp_layers = [
            nn.ConvTranspose2d(input_dimension, conv_dimension * multiplier, kernel_size=4, stride=1,padding=0, bias=False),
            nn.BatchNorm2d(conv_dimension * multiplier),
            nn.ReLU()
        ]

        for layer_index in range(used_layers):
            multiplier = int(multiplier // 2)
            temp_layers+=[
                nn.ConvTranspose2d(conv_dimension * (multiplier * 2), conv_dimension * multiplier, kernel_size=4, stride=2,padding=1, bias=True),
                nn.BatchNorm2d(conv_dimension * multiplier),
                nn.ReLU()
            ]

        temp_layers+=[
            nn.ConvTranspose2d(conv_dimension * multiplier, self.image_shape[0], kernel_size=3, stride=1,padding=1, bias=False),
            nn.Tanh()
        ]

        self.layers = nn.Sequential(*temp_layers)

        self.apply(init_weight)

    def generate_embedding(self, labels: torch.tensor):
        return self.label_embedding(labels.float()).unsqueeze(-1).unsqueeze(-1)

    def forward(self, latent_vector: torch.tensor, labels: torch.tensor = None):
        final_latent_vector = latent_vector
        if self.conditional:
            embedded_labels = self.generate_embedding(labels)

            final_latent_vector = torch.cat([latent_vector, embedded_labels], dim=1)

        return self.layers(final_latent_vector)
    
    def forward_without_embedding(self, latent_vector: torch.tensor, embedding_vector: torch.tensor):
        final_latent_vector = latent_vector
        if self.conditional:
            final_latent_vector = torch.cat([latent_vector, embedding_vector], dim=1)
        return self.layers(final_latent_vector)
    
    def load_model_state(self, model_state):
        new_state = OrderedDict()
        kept_layers = 0

        current_model_state = self.state_dict()
        for X,Y in current_model_state.items():
            if X in model_state.keys() and model_state[X].shape == Y.shape:
                new_state[X] = model_state[X]
                if "layers" in X:
                    kept_layers = int(X.split(".")[1])
            else:
                new_state[X] = Y

        total_layers = int([l for l in current_model_state.keys() if "layers" in l][-1].split(".")[1])

        self.load_state_dict(new_state)
        return total_layers - kept_layers, total_layers
    
    def generate_latents(self, batch_size: int, device: str) -> torch.tensor:
        return torch.randn((batch_size, self.latent_dimension, 1, 1), device=device, dtype=torch.float32)
    
    def generate_labels(self, batch_size: int, device: str) -> torch.tensor:
        return torch.randint(0, 2, size=(batch_size,self.num_labels,), device=device)

    @torch.no_grad()
    def requires_gradients(self, layer_numbers: int, state: bool):
        for index in range(layer_numbers):
            for param in self.layers[index].parameters():
                param.requires_grad_(state)
    
class WDiscriminator(BaseDiscriminator):
    def __init__(self, image_shape: tuple, used_layers: int, total_layers: int, conv_dimension: int = 48, num_labels: int = 0):
        super(WDiscriminator, self).__init__()
        self.image_shape = image_shape
        self.conv_dimension = conv_dimension
        self.used_layers = used_layers
        self.total_layers = total_layers

        self.num_labels = num_labels
        self.conditional = num_labels > 0

        input_dimension = self.image_shape[0]
        if self.conditional:
            self.label_embedding = nn.Sequential(nn.Linear(self.num_labels, self.num_labels * 2), nn.Linear(self.num_labels * 2, self.image_shape[1] * self.image_shape[2]))
            input_dimension +=1

        multiplier = 1

        temp_layers = [
            nn.Conv2d(input_dimension, conv_dimension * multiplier, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dimension * multiplier),
            nn.LeakyReLU(0.2),
        ]

        for layer_index in range(used_layers):
            multiplier = multiplier * 2
            temp_layers+=[
                torch.nn.utils.parametrizations.spectral_norm(nn.Conv2d(conv_dimension * int(multiplier // 2), conv_dimension * multiplier, kernel_size=4, stride=2,padding=1, bias=False)),
                nn.BatchNorm2d(conv_dimension * multiplier),
                nn.LeakyReLU(0.2),
            ]

        temp_layers+=[
            torch.nn.utils.parametrizations.spectral_norm(nn.Conv2d(conv_dimension * multiplier, 1, kernel_size=2, stride=1,padding=0, bias=False))
        ]

        self.layers = nn.Sequential(*temp_layers)

        self.apply(init_weight)

    def generate_embedding(self, labels: torch.tensor):
        return self.label_embedding(labels.float()).view(labels.size(0), 1, self.image_shape[1], self.image_shape[2])

    def forward(self, image_tensor: torch.tensor, labels: torch.tensor = None):
        final_image_tensor = image_tensor
        if self.conditional:
            embedded_labels = self.generate_embedding(labels)
            final_image_tensor = torch.cat([image_tensor,embedded_labels], dim=1)

        score = self.layers(final_image_tensor)
        return score.view(image_tensor.shape[0],1)
    
    def forward_without_embedding(self, latent_vector: torch.tensor, embedding_vector: torch.tensor):
        final_latent_vector = latent_vector
        if self.conditional:
            final_latent_vector = torch.cat([latent_vector, embedding_vector], dim=1)
        return self.layers(final_latent_vector)

    def load_model_state(self, model_state):
        new_state = OrderedDict()
        kept_layers = 0

        current_model_state = self.state_dict()
        for X,Y in current_model_state.items():
            if X in model_state.keys() and model_state[X].shape == Y.shape:
                new_state[X] = model_state[X]
                if "layers" in X:
                    kept_layers = int(X.split(".")[1])
            else:
                new_state[X] = Y

        total_layers = int([l for l in current_model_state.keys() if "layers" in l][-1].split(".")[1])

        self.load_state_dict(new_state)
        return total_layers - kept_layers, total_layers
    
    @torch.no_grad()
    def requires_gradients(self, layer_numbers: int, state: bool):
        for index in range(layer_numbers):
            for param in self.layers[index].parameters():
                param.requires_grad_(state)