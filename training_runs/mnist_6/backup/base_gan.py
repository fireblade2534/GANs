import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseGenerator(nn.Module, ABC):
    @abstractmethod
    def forward(self, latent_vector: torch.tensor):
        raise NotImplementedError("Subclasses must implement the forward method")
    
    @abstractmethod
    def generate_latents(self, batch_size: int, device: str) -> torch.tensor:
        raise NotImplementedError("Subclasses must implement the generate_latents method")

    @abstractmethod
    def load_model_state(self, model_state):
        raise NotImplementedError("Subclasses must implement the load_model_state method")
    
    @abstractmethod
    def requires_gradients(self, layer_numbers: int, state: bool):
        raise NotImplementedError("Subclasses must implement the requires_gradients method")

class BaseDiscriminator(nn.Module, ABC):
    @abstractmethod
    def forward(self, image_tensor: torch.tensor):
        raise NotImplementedError("Subclasses must implement the forward method")
    
    @abstractmethod
    def load_model_state(self, model_state):
        raise NotImplementedError("Subclasses must implement the load_model_state method")
    
    @abstractmethod
    def requires_gradients(self, layer_numbers: int, state: bool):
        raise NotImplementedError("Subclasses must implement the requires_gradients method")