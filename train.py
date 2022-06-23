from pathlib import Path

import torch
import pytorch_lightning as pl
from deep_utils import mkdir_incremental, BlocksTorch, get_logger, log_print
from settings import Config
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_model import LitModel
from dataset import ImageClassificationDataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", default="squeezenet", help="model name to be trained, default is squeezenet")
parser.add_argument("--dataset_dir", type=Path, default="./dataset", help="path to the dataset, default: ./dataset")
parser.add_argument("--output_dir", type=Path, default="./output",
                    help="path to the output directory, default: ./output")
parser.add_argument("--train_epochs", type=int, default=5, help="number of training epochs")
parser.add_argument("--finetune_epochs", type=int, default=5, help="number of fine-tuning epochs")
parser.add_argument("--device", default=None,
                    help="what should be the device for training, default is None! If cuda is found it will be chosen!")
parser.add_argument("--n_workers", type=int, default=8, help="Number of workers for data-loaders")
parser.add_argument("--finetune_layers", type=int, default=50,
                    help="Number of layers that should be finetuned starting from the end of the layers. Default is 50")


def main():
    config = Config()
    args = parser.parse_args()
    config.update_config_param(args)
    output_dir = mkdir_incremental(config.output_dir)
    train_dir = output_dir / "train"
    finetune_dir = output_dir / "finetune"
    logger = get_logger("pytorch-lightning-image-classification", log_path=output_dir / "log.log")
    log_print(logger, f"Config files: {config}")
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       filename=config.file_name,
                                       monitor="val_loss",
                                       verbose=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(gpus=1 if config.device == "cuda" else 0,
                         max_epochs=config.train_epochs,
                         min_epochs=config.train_epochs // 10,
                         callbacks=[model_checkpoint, learning_rate_monitor],
                         default_root_dir=train_dir)
    lit_model = LitModel(config.model_name, config.n_classes, config.last_layer_nodes, config.train_lr,
                         config.lr_reduce_factor, config.lr_patience)
    lit_model.model.model_ft.classifier[1].apply(BlocksTorch.weights_init)
    train_loader, val_loader = ImageClassificationDataset.get_loaders(config)
    log_print(logger, "Training the model...")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    log_print(logger, "Fine-tuning the model...")
    lit_model.lr = config.finetune_lr
    BlocksTorch.set_parameter_grad(lit_model, config.finetune_layers)

    trainer = pl.Trainer(gpus=1 if config.device == "cuda" else 0,
                         max_epochs=config.finetune_epochs,
                         min_epochs=config.finetune_epochs // 10,
                         callbacks=[model_checkpoint, learning_rate_monitor],
                         default_root_dir=finetune_dir)
    trainer.fit(model=lit_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    log_print(logger, "Testing val_loader...")
    trainer.test(lit_model, ckpt_path="best", dataloaders=val_loader)

    log_print(logger, "Testing train_loader...")
    trainer.test(lit_model, ckpt_path="best", dataloaders=train_loader)

    # Adding artifacts to weights
    weight_path = output_dir / f"{config.file_name}.ckpt"
    best_weight = torch.load(weight_path)
    best_weight['id_to_class'] = {v: k for k, v in train_loader.dataset.class_to_id.items()}
    for k, v in config.vars().items():
        if k not in best_weight:
            best_weight[k] = v
        else:
            log_print(logger, f"[Warning] Did not save {k} = {v} because there is a variable with the same name!")
    if config.save_model_w_weight:
        best_weight['model'] = lit_model.model
    torch.save(best_weight, weight_path)


if __name__ == '__main__':
    main()
