from typing import Any, Dict, List
import matplotlib.pyplot as plt
import torch
import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from src.objects.training_state import TrainState
from src.objects.training_config import TrainingConfig
from src.base.base_gan import BaseGenerator, BaseDiscriminator
import os
import yaml
from dataclasses import asdict
import torchvision.transforms as transforms
from src.logging import *
from src.utilities.diff_agument import DiffAugment

class BaseTrainer(ABC):
    @abstractmethod
    def __init__(self, generator: BaseGenerator, discriminator: BaseDiscriminator, generator_optimizer: Optimizer, discriminator_optimizer: Optimizer, generator_scheduler: LRScheduler, discriminator_scheduler: LRScheduler, device: str = "cpu"):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.device = device

    @abstractmethod
    def _train_generator_iteration(self, zero_grad: bool):
        raise NotImplementedError("Subclasses must implement the _train_generator_iteration method")
    
    @abstractmethod
    def _train_discriminator_iteration(self, real_images: torch.tensor, fake_images: torch.tensor, zero_grad: bool, real_labels: torch.tensor = None, fake_labels: torch.tensor = None):
        raise NotImplementedError("Subclasses must implement the _train_discriminator_iteration method")

    @abstractmethod
    def _on_train_start(self):
        raise NotImplementedError("Subclasses must implement the _on_train_start method")

    @classmethod
    def _convert_dataset_to_dict(cls, dataset: Dataset):
        output_dict = {"args": {}}
        
        if isinstance(dataset, datasets.ImageFolder):
            output_dict["type"] = "ImageFolder"
            output_dict["args"]["root"] = dataset.root
            output_dict["args"]["transform"] = cls._convert_transforms_to_dict(dataset.transform)

        elif isinstance(dataset, datasets.MNIST):
            output_dict["type"] = "MNIST"
            output_dict["args"]["root"] = dataset.root
            output_dict["args"]["train"] = dataset.train
            output_dict["args"]["transform"] = cls._convert_transforms_to_dict(dataset.transform)

        else:
            logger.Critical(f"Failed to seralize dataset: {dataset}. Exiting")
            exit()

        return output_dict
        
    @classmethod
    def _convert_dict_to_dataset(cls, dataset_dict: Dict[str, Any]):
        dataset_function = None
        dataset_args = dataset_dict["args"]
        
        if dataset_dict["type"] == "ImageFolder":
            dataset_function = datasets.ImageFolder
            dataset_args["transform"] = cls._convert_dict_to_transforms(dataset_args["transform"])

        elif dataset_dict["type"] == "MNIST":
            dataset_function = datasets.MNIST
            dataset_args["transform"] = cls._convert_dict_to_transforms(dataset_args["transform"])
        
        else:
            logger.Critical(f"Failed to parse: {dataset_dict['type']}. Exiting")
            exit()

        return dataset_function(**dataset_args)


    @classmethod
    def _convert_transforms_to_dict(cls, transform_compose: transforms.Compose):
        output_list = []

        for transform in transform_compose.transforms:
            if isinstance(transform, transforms.ToTensor):
                output_list.append({"type": "ToTensor", "args": {}})

            elif isinstance(transform, transforms.Grayscale):
                output_list.append({"type": "Greyscale", "args": {"num_output_channels": transform.num_output_channels}})

            elif isinstance(transform, transforms.Resize):
                output_list.append({"type": "Resize", "args": {"size": list(transform.size), "max_size": transform.max_size, "antialias": transform.antialias, "interpolation": transform.interpolation.value}})

            elif isinstance(transform, transforms.Normalize):
                output_list.append({"type": "Normalize", "args": {"mean": transform.mean, "std": transform.std, "inplace": transform.inplace}})

            else:
                logger.Critical(f"Failed to seralize transform: {transform}. Exiting")
                exit()
        return output_list
    
    @classmethod
    def _convert_dict_to_transforms(cls, transform_dict: List[Dict[str, Any]]):
        output_compose = []

        for transform in transform_dict:
            transform_function = None
            transform_args = transform["args"]

            if transform["type"] == "ToTensor":
                transform_function = transforms.ToTensor

            elif transform["type"] == "Greyscale":
                transform_function = transforms.Grayscale

            elif transform["type"] == "Resize":
                transform_function = transforms.Resize
                transform_args["size"] = tuple(transform_args["size"])
                transform_args["interpolation"] = getattr(transforms.InterpolationMode, transform_args["interpolation"])

            elif transform["type"] == "Normalize":
                transform_function = transforms.Normalize

            else:
                logger.Critical(f"Failed to parse: {transform['type']}. Exiting")
                exit()

            output_compose.append(transform_function(**transform_args))

        return transforms.Compose(output_compose)

    @classmethod
    def _save_training_config(
        cls,
        name: str,
        save_path: str,
        training_config: TrainingConfig,
        dataset: Dataset,
        resume_path: str = None,
        override_resume_options: bool = False):

        training_config_dict = asdict(training_config)
        dataset_config_dict = cls._convert_dataset_to_dict(dataset)    

        yaml_str = yaml.dump({"name": name, "save_path": save_path, "training_config": training_config_dict, "dataset": dataset_config_dict, "resume_path": resume_path, "override_resume_options": override_resume_options}, sort_keys=False)

        with open(os.path.join(save_path, "config.yaml"), "w") as config_file:
            config_file.write(yaml_str)

    def save_checkpoint(self, epoch: int, checkpoint_path: str):
        generator_to_save = self.generator._orig_mod if hasattr(self.generator, '_orig_mod') else self.generator
        generator_to_save = generator_to_save.to("cpu").state_dict()

        discriminator_to_save = self.discriminator._orig_mod if hasattr(self.discriminator, '_orig_mod') else self.discriminator
        discriminator_to_save = discriminator_to_save.to("cpu").state_dict()

        torch.save({
            "name": self.name,
            "train_state": TrainState(epoch=epoch, generator=generator_to_save, discriminator=discriminator_to_save, generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer, generator_scheduler=self.generator_scheduler, discriminator_scheduler=self.discriminator_scheduler),
        }, checkpoint_path)

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> TrainState:
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        return checkpoint.get("train_state", None)

    def train_from_config(self, config_path: str):
        with open(config_path, "r") as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

        train_args = config_dict

        train_args["training_config"] = TrainingConfig(**train_args["training_config"])
        train_args["dataset"] = self._convert_dict_to_dataset(train_args["dataset"])

        self.train(**train_args)

    def train(
        self,
        name: str,
        save_path: str,
        training_config: TrainingConfig,
        dataset: Dataset,
        resume_path: str = None,
        override_resume_options: bool = False
    ):
        self.name = name
        self.training_config = training_config

        self.conditional = self.training_config.num_labels > 0
        
        generator_freeze_layers = 0
        discriminator_freeze_layers = 0
        start_epoch = 1

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

        torch.manual_seed(training_config.seed)        

        logger.Info("Saving training config")
        self._save_training_config(name, save_path, training_config, dataset, resume_path, override_resume_options)
        
        if resume_path is not None:
            if not os.path.isfile(resume_path):
                logger.Critical(f"No checkpoint file found at {resume_path}. Aborting")
                exit()

            logger.Info(f"Resuming training from checkpoint at {resume_path}")
            training_state = self.load_checkpoint(resume_path)

            start_epoch = training_state.epoch + 1

            generator_modified_layers, generator_total_layers = self.generator.load_model_state(training_state.generator)
            discriminator_modified_layers, discriminator_total_layers = self.discriminator.load_model_state(training_state.discriminator)

            self.generator.requires_gradients(generator_total_layers, True)
            self.discriminator.requires_gradients(discriminator_total_layers, True)

            logger.Info("Loaded generator and discriminator from checkpoint")

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        logger.Debug(f"Transfered the generator and discriminator to {self.device}")
        
        if resume_path is not None:
            if not override_resume_options:
                self.generator_optimizer.load_state_dict(training_state.generator_optimizer.state_dict())
                self.discriminator_optimizer.load_state_dict(training_state.discriminator_optimizer.state_dict())
                logger.Info("Loaded generator and discriminator optimizer from checkpoint")

                self.generator_scheduler.load_state_dict(training_state.generator_scheduler.state_dict())
                self.discriminator_scheduler.load_state_dict(training_state.discriminator_scheduler.state_dict())
                logger.Info("Loaded generator and discriminator scheduler from checkpoint")

            if generator_modified_layers != 0:
                generator_freeze_layers = generator_total_layers - generator_modified_layers
                self.generator.requires_gradients(generator_freeze_layers, False)
                logger.Info(f"Freezing the first {generator_freeze_layers} generator layers")

            if discriminator_modified_layers != 0:
                discriminator_freeze_layers = discriminator_total_layers - discriminator_modified_layers
                self.discriminator.requires_gradients(discriminator_freeze_layers, False)
                logger.Info(f"Freezing the first {discriminator_freeze_layers} discriminator layers")

        self.generator.train(True)
        self.discriminator.train(True)

        self.diff_augment = DiffAugment(self.training_config.augmentation_config, image_shape=self.generator.image_shape, device=self.device)

        if self.device == "cuda":
            self.generator = torch.compile(self.generator, mode="default")
            #self.discriminator = torch.compile(self.discriminator, mode="default")

        dataloader = DataLoader(  # torch.utils.data.DataLoader
            dataset, batch_size=training_config.batch_size, shuffle=True, drop_last=True, num_workers=training_config.num_data_workers, pin_memory=self.device == "cuda"
        )

        total_batches = len(dataloader)

        test_images = training_config.sample_grid_size * training_config.sample_grid_size

        test_latents = self.generator.generate_latents((test_images), device=self.device)
        test_labels = torch.arange(0, self.training_config.num_labels, device=self.device).repeat(test_images // self.training_config.num_labels + 1)[:test_images]

        self._on_train_start()

        logger.Info(f"Starting training at epoch {start_epoch} with settings:")
        logger.Info(f"  Latent Dim: {self.generator.latent_dimension}")
        logger.Info(f"  Generator Learning Rate: {self.generator_optimizer.param_groups[0]['lr']:.6f}")
        logger.Info(f"  Discriminator Learning Rate: {self.discriminator_optimizer.param_groups[0]['lr']:.6f}")
        logger.Info(f"  Batch Size: {training_config.batch_size}")
        logger.Info(f"  Epochs: {training_config.epochs}")
        logger.Info(f"  Device: {self.device}")
        logger.Info(f"  Workers: {training_config.num_data_workers}")
        
        discriminator_steps_taken = 0
        generator_steps_taken = 0
        for epoch in range(start_epoch, training_config.epochs + 1):
            if (epoch - start_epoch) >= training_config.stablization_epochs:
                if generator_freeze_layers > 0:
                    self.generator.requires_gradients(generator_freeze_layers, True)
                    logger.Info(f"Thawed the first {generator_freeze_layers} generator layers")
                    generator_freeze_layers = 0
            
                if discriminator_freeze_layers > 0:
                    self.discriminator.requires_gradients(discriminator_freeze_layers, True)
                    logger.Info(f"Thawed the first {discriminator_freeze_layers} discriminator layers")
                    discriminator_freeze_layers = 0

            generator_epoch_loss_history = {}
            discriminator_epoch_loss_history = {}

            for batch_index, (real_images, real_labels) in enumerate(tqdm.tqdm(dataloader, unit="batch")):
                torch.compiler.cudagraph_mark_step_begin()


                for discriminator_iterations in range(0, training_config.discriminator_repeats):
                    discriminator_steps_taken+=1
                    
                    fake_latent_vector = self.generator.generate_latents(self.training_config.batch_size, device=self.device)
                    fake_labels = self.generator.generate_labels(self.training_config.batch_size, device=self.device)

                    fake_images = self.generator(fake_latent_vector, fake_labels)

                    should_zero_grad = discriminator_steps_taken % self.training_config.gradient_accumulation_steps == 0

                    discriminator_losses = self._train_discriminator_iteration(real_images.to(device=self.device), fake_images.detach(), should_zero_grad, real_labels.to(device=self.device), fake_labels.detach())

                    for key in discriminator_losses:
                        if key in discriminator_epoch_loss_history:
                            discriminator_epoch_loss_history[key] += discriminator_losses[key] / (self.training_config.discriminator_repeats * total_batches)
                        else:
                            discriminator_epoch_loss_history[key] = discriminator_losses[key] / (self.training_config.discriminator_repeats * total_batches)

                generator_steps_taken+=1
                    
                should_zero_grad = generator_steps_taken % training_config.gradient_accumulation_steps == 0

                generator_losses = self._train_generator_iteration(should_zero_grad)
                
                for key in generator_losses:
                    if key in generator_epoch_loss_history:
                        generator_epoch_loss_history[key] += generator_losses[key] / total_batches
                    else:
                        generator_epoch_loss_history[key] = generator_losses[key] / total_batches

            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            self.generator.eval()
            self.discriminator.eval()

            with torch.no_grad():
                if epoch % training_config.sample_epochs == 0 or epoch == training_config.epochs:
                    sample_visualizations = self.generator(test_latents.detach(), test_labels.detach()).cpu().permute(0, 2, 3, 1)
                    #sample_visualizations = self.diff_augment.apply_agumentation(real_images.cuda()).cpu().permute(0, 2, 3, 1)

                    _, axes = plt.subplots(nrows=training_config.sample_grid_size, ncols=training_config.sample_grid_size, figsize=(training_config.sample_grid_size + 1, training_config.sample_grid_size + 1))
                    plt.suptitle(f"EPOCH : {epoch}")

                    for index in range(training_config.sample_grid_size*training_config.sample_grid_size):
                        axes.flat[index].imshow((sample_visualizations[index] + 1) / 2)#, cmap="gray")
                        axes.flat[index].axis("off")

                    plt.savefig(os.path.join(save_path, f"samples/{name}_{epoch}_sample.png"))
                    plt.close()

                if epoch % training_config.save_epochs == 0 or epoch == training_config.epochs:
                    self.save_checkpoint(epoch, os.path.join(save_path, f"checkpoints/{name}_{epoch}_model.pt"))
            self.generator.to(device=self.device)
            self.discriminator.to(device=self.device)

            self.generator.train(True)
            self.discriminator.train(True)

            logger.Info(f"Finished epoch {epoch}/{training_config.epochs}:")
            for key_name, loss in generator_epoch_loss_history.items():
                logger.Info(f"  {key_name}: {loss:.6f}")

            for key_name, loss in discriminator_epoch_loss_history.items():
                logger.Info(f"  {key_name}: {loss:.6f}")

            logger.Info(f"  Generator Learning Rate: {self.generator_optimizer.param_groups[0]['lr']:.6f}")
            logger.Info(f"  Discriminator Learning Rate: {self.discriminator_optimizer.param_groups[0]['lr']:.6f}")
            logger.Info(f"-----------------------------------------------------")
