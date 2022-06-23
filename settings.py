import os
from dataclasses import dataclass
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass(init=True, repr=True)
class DirConfig:
    dataset_dir = "./dataset"
    output_dir = "./output"
    file_name = "best"


@dataclass(init=True, repr=True)
class ModelConfig:
    model_name = "squeezenet"
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    last_layer_nodes = 512

    n_classes = len(
        [d for d in os.listdir(DirConfig.dataset_dir) if
         os.path.isdir(os.path.join(DirConfig.dataset_dir, d))]) if os.path.isdir(DirConfig.dataset_dir) else 0


@dataclass(init=True, repr=True)
class AugConfig:
    train_transform = A.Compose(
        [A.Resize(height=ModelConfig.input_size, width=ModelConfig.input_size),
         A.Rotate(limit=20, p=0.2),
         A.HorizontalFlip(p=0.5),
         A.Normalize(ModelConfig.mean, ModelConfig.std, max_pixel_value=255.0),
         ToTensorV2()
         ])
    val_transform = A.Compose(
        [A.Resize(ModelConfig.input_size, ModelConfig.input_size),
         A.Normalize(ModelConfig.mean, ModelConfig.std, max_pixel_value=255.0),
         ToTensorV2()
         ])


@dataclass(init=True, repr=True)
class DeviceConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if torch.cuda.is_available() else False
    n_workers = 8

    @classmethod
    def update_device(cls):
        if cls.device is None:
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.pin_memory = True if cls.device != 'cpu' else False


@dataclass(init=True, repr=True)
class SaveConfig:
    save_model_w_weight = True


@dataclass(init=True, repr=True)
class Config(DirConfig, ModelConfig, AugConfig, DeviceConfig, SaveConfig):
    train_epochs = 5
    train_lr = 1e-3

    finetune_epochs = 5
    finetune_lr = 1e-4
    finetune_layers = 50

    lr_reduce_factor = 0.1
    lr_patience = 5

    validation_size = 0.2
    batch_size = 64

    def update_config_param(self, args):
        variables = vars(args)
        for k, v in variables.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"value {k} is not defined in Config...")
        self.update()

    def update_model(self):
        self.n_classes = len(
            [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))])

    def update(self):
        self.update_device()
        self.update_model()

    def __repr__(self):
        variables = vars(self)
        return f"{self.__class__.__name__} -> " + ", ".join(f"{k}: {v}" for k, v in variables.items())
