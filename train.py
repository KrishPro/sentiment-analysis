"""
Written by KrishPro @ KP
"""

from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from data import IMDBDataModule
from model import Model

class TrainModel(Model):
    def __init__(self, learning_rate: float, ultimate_batch_size: int, epochs: int):
        super(TrainModel, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.total_steps = (45_000 // ultimate_batch_size) * epochs
        self.warmup_steps = int(0.1 * self.total_steps)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int):
        (input_ids, attention_mask), label = batch
        preds = self(input_ids, attention_mask)
        loss = self.criterion(preds, torch.abs(label - 0.1))
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("lr", self.optimizers().optimizer.param_groups[0]['lr'], prog_bar=True)
        return loss
    

    def validation_step(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int):
        (input_ids, attention_mask), label = batch
        preds = self(input_ids, attention_mask)
        loss = self.criterion(preds, label)
        self.log("val_loss", loss.item(), prog_bar=True)
        return loss

    def test_step(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int):
        (input_ids, attention_mask), label = batch
        preds = self(input_ids, attention_mask)
        loss = self.criterion(preds, label)
        return loss

if __name__ == '__main__':
    # epochs = 50
    # trainer = Trainer(tpu_cores=8, max_epochs=epochs)
    # datamodule = IMDBDataModule(batch_size=16)
    # model = TrainModel(learning_rate=2e-5, ultimate_batch_size=8*16, epochs=epochs)
    # trainer.fit(model, datamodule)

    pass