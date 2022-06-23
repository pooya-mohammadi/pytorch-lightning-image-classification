import torch
import pytorch_lightning as pl
from deep_utils import log_print
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from deep_utils import TorchVisionModel
from torch import nn


class LitModel(pl.LightningModule):
    def __init__(self, model_name, n_classes, last_layer_nodes, lr, lr_reduce_factor, lr_patience, logger=None,
                 verbose=1):
        super(LitModel, self).__init__()
        self.save_hyperparameters()

        self.lr = self.hparams.lr
        self.model = TorchVisionModel(model_name=self.hparams.model_name,
                                      num_classes=self.hparams.n_classes,
                                      last_layer_nodes=self.hparams.last_layer_nodes)
        self.criterion = nn.CrossEntropyLoss()
        self.outer_logger = logger
        self.verbose = verbose
        self.epoch = 0

    def forward(self, x):
        logit = self.model(x)
        return logit

    def training_step(self, batch, batch_idx):
        return self.get_loss_acc(batch)

    def test_step(self, batch, batch_idx):
        return self.get_loss_acc(batch)

    def test_epoch_end(self, outputs) -> None:
        self.calculate_metrics(outputs, type_="test")

    def validation_epoch_end(self, outputs) -> None:
        self.calculate_metrics(outputs, type_="val")

    def training_epoch_end(self, outputs) -> None:
        self.calculate_metrics(outputs, type_='train')
        self.epoch += 1

    def calculate_metrics(self, outputs, type_="train"):
        labels, preds = [], []
        r_acc, r_loss, size = 0, 0, 0
        for row in outputs:
            r_acc += row["acc"]
            r_loss += row["loss"]
            size += row["bs"]
            preds.extend(row['preds'])
            labels.extend(row["labels"])
        f1_value = f1_score(labels, preds, average="weighted")
        loss = r_loss / size
        acc = r_acc / size
        log_print(self.outer_logger,
                  f"Epoch: {self.epoch} - {type_}-acc: {acc} - {type_}-loss: {loss} - {type_}-f1-score: {f1_value}")
        self.log(f"{type_}_f1_score", f1_value)
        self.log(f"{type_}_loss", loss.item())
        self.log(f"{type_}_acc", acc, )
        return acc, f1_value, loss

    def validation_step(self, batch, batch_idx):
        return self.get_loss_acc(batch)

    def get_loss_acc(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        logits = self.model(images)
        loss = self.criterion(logits, labels) * batch_size
        _, preds = torch.max(logits, 1)
        corrects = torch.sum(preds == labels.data)
        return {"acc": corrects.item(),
                "loss": loss,
                "bs": batch_size,
                "preds": preds.cpu().numpy().tolist(),
                "labels": labels.cpu().numpy().tolist()
                }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_reduce_factor,
                                      patience=self.hparams.lr_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
