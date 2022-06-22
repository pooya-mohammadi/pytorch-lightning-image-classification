import torch
import pytorch_lightning as pl
from deep_utils import mkdir_incremental, BlocksTorch, get_logger, log_print
from settings import Config
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_model import LitModel


def main():
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
    train_loader, val_loader = lit_model.get_loaders(output_dir)
    log_print(logger, "Training the model...")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    log_print(logger, "Fine-tuning the model...")
    lit_model.lr = Config.finetune_lr
    lit_model.model.set_parameter_grad(Config.finetune_layers)

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
    weight_path = output_dir / f"{Config.FILE_NAME}.ckpt"
    best_weight = torch.load(weight_path)
    best_weight['Config'] = Config
    torch.save(best_weight, weight_path)


if __name__ == '__main__':
    main()
