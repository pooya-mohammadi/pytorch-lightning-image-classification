import torch
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TorchVisionModel
from settings import Config
from torch import nn


class LitModel(pl.LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.lr = Config.train_lr
        self.model = TorchVisionModel(model_name=Config.model_name, num_classes=Config.n_classes,
                                      last_layer_nodes=Config.last_layer_nodes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logit = self.model(x)
        return logit

    def training_step(self, batch, batch_idx):
        acc, loss, bs, preds, labels = self.get_loss_acc(batch)
        return {"acc": acc, "loss": loss, "bs": bs, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        acc, loss, bs, preds, labels = self.get_loss_acc(batch)
        return {"acc": acc, "loss": loss, "bs": bs, "preds": preds, "labels": labels}

    def test_epoch_end(self, outputs) -> None:
        acc, f1_value, loss = self.calculate_metrics(outputs)
        self.log("test_f1_score", f1_value)
        self.log("test_loss", loss.item())
        self.log("test_acc", acc)

    def validation_epoch_end(self, outputs) -> None:
        acc, f1_value, loss = self.calculate_metrics(outputs)
        self.log("val_f1_score", f1_value)
        self.log("val_loss", loss.item())
        self.log("val_acc", acc)

    @staticmethod
    def calculate_metrics(outputs):
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
        return acc, f1_value, loss

    def training_epoch_end(self, outputs) -> None:
        acc, f1_value, loss = self.calculate_metrics(outputs)
        self.log("train_f1_score", f1_value)
        self.log("train_loss", loss.item())
        self.log("train_acc", acc, )

    def validation_step(self, batch, batch_idx):
        acc, loss, bs, preds, labels = self.get_loss_acc(batch)
        return {"acc": acc, "loss": loss, "bs": bs, "preds": preds, "labels": labels}

    def get_loss_acc(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        logits = self.model(images)
        loss = self.criterion(logits, labels) * batch_size
        _, preds = torch.max(logits, 1)
        corrects = torch.sum(preds == labels.data)
        return corrects.item(), loss, batch_size, preds.cpu().numpy().tolist(), labels.cpu().numpy().tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.lr_reduce_factor,
                                      patience=Config.lr_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
