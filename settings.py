import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DirConfig:
    dataset_dir = "./dataset"
    output_dir = "./output"
    file_name = "best"


class ModelConfig:
    model_name = "squeezenet"
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    last_layer_nodes = 512


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


class DeviceConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if torch.cuda.is_available() else False
    n_workers = 8


class Config(DirConfig, ModelConfig, AugConfig, DeviceConfig):
    train_epochs = 25
    train_lr = 1e-3

    finetune_epochs = 50
    finetune_lr = 1e-4
    finetune_layers = 50

    lr_reduce_factor = 0.1
    lr_patience = 5

    validation_size = 0.2
    batch_size = 64

    n_classes = len(
        [d for d in os.listdir(DirConfig.dataset_dir) if os.path.isdir(os.path.join(DirConfig.dataset_dir, d))])
