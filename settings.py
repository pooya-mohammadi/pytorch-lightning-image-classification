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

    n_classes = len(
        [d for d in os.listdir(DirConfig.dataset_dir) if
         os.path.isdir(os.path.join(DirConfig.dataset_dir, d))]) if os.path.isdir(DirConfig.dataset_dir) else 0


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

    @classmethod
    def update_device(cls):
        if cls.device is None:
            cls.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.pin_memory = True if cls.device != 'cpu' else False


class SaveConfig:
    save_model_w_weight = True


class Config(DirConfig, ModelConfig, AugConfig, DeviceConfig, SaveConfig):
    train_epochs = 15
    train_lr = 1e-3

    finetune_epochs = 25
    finetune_lr = 1e-4
    finetune_layers = 50

    lr_reduce_factor = 0.1
    lr_patience = 5

    validation_size = 0.2
    batch_size = 64

    @classmethod
    def update_config_param(cls, args):
        variables = vars(args)
        for k, v in variables.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                raise ValueError(f"value {k} is not defined in Config...")
        cls.update()

    @classmethod
    def update_model(cls):
        cls.n_classes = len(
            [d for d in os.listdir(cls.dataset_dir) if os.path.isdir(os.path.join(cls.dataset_dir, d))])

    @classmethod
    def update(cls):
        cls.update_device()
        cls.update_model()
