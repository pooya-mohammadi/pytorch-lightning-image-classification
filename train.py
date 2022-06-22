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
parser.add_argument("--dataset_dir", default="./dataset", help="path to the dataset, default: ./dataset")
parser.add_argument("--output_dir", default="./output", help="path to the output directory, default: ./output")
parser.add_argument("--train_epochs", default=15, help="number of training epochs")
parser.add_argument("--finetune_epochs", default=25, help="number of fine-tuning epochs")
parser.add_argument("--device", default=None,
                    help="what should be the device for training, default is None! If cuda is found it will be chosen!")


def main():
    args = parser.parse_args()
    Config.update_config_param(args)
    output_dir = mkdir_incremental(Config.output_dir)
    train_dir = output_dir / "train"
    finetune_dir = output_dir / "finetune"
    logger = get_logger("pytorch-lightning-image-classification", log_path=output_dir / "log.log")
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       filename=Config.file_name,
                                       monitor="val_loss",
                                       verbose=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(gpus=1 if Config.device == "cuda" else 0,
                         max_epochs=Config.train_epochs,
                         min_epochs=Config.train_epochs // 10,
                         callbacks=[model_checkpoint, learning_rate_monitor],
                         default_root_dir=train_dir)
    lit_model = LitModel()
    lit_model.model.model_ft.classifier[1].apply(BlocksTorch.weights_init)
    train_loader, val_loader = ImageClassificationDataset.get_loaders()
    log_print(logger, "Training the model...")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    log_print(logger, "Fine-tuning the model...")
    lit_model.lr = Config.finetune_lr
    BlocksTorch.set_parameter_grad(lit_model, Config.finetune_layers)

    trainer = pl.Trainer(gpus=1 if Config.device == "cuda" else 0,
                         max_epochs=Config.finetune_epochs,
                         min_epochs=Config.finetune_epochs // 10,
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
    weight_path = output_dir / f"{Config.file_name}.ckpt"
    best_weight = torch.load(weight_path)
    best_weight['Config'] = Config
    best_weight['class_to_id'] = train_loader.dataset.class_to_id
    if Config.save_model_w_weight:
        best_weight['model'] = lit_model.model
    torch.save(best_weight, weight_path)


if __name__ == '__main__':
    main()
