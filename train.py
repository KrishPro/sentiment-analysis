"""
Written by KrishPro @ KP
"""

from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import LearningRateMonitor
from pyngrok import ngrok
from tensorboard import program
from data import IMDBDataModule
from model import Model

# # Setup Colab
# # Installing torch/xla (TPU), pytorch_lightning, transformers & pyngrok
# !pip install -q torchtext==0.10 torchaudio==0.9.0 torchvision==0.10.0 tf-estimator-nightly==2.8.0.dev2021122109 earthengine-api==0.1.238 folium==0.2.1
# !pip install -q cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
# !pip install -q pytorch_lightning transformers pyngrok
# # Registering AuthToken for ngrok
# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.tgz
# !tar zxvf ngrok-stable-linux-amd64.tgz
# !./ngrok authtoken 1y2MQMr0xLh05Dvbb0dABiNQpAY_3bqEfwwEtM7duDaqwrN93
# # Registering username & key for kaggle
# !echo '{"username":"krishbaisoya","key":"d88e0b0654476a2d36bd04dd95bf2fb4"}' > kaggle.json
# !mkdir ~/.kaggle
# !mv kaggle.json ~/.kaggle/
# # Mounting Drive
# from google.colab import drive
# drive.mount('/content/drive')

def start_tensorboard(log_dir: str = "/content/drive/MyDrive/Models/sentiment-analysis/fine-tuning/lightning_logs"):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    print(ngrok.connect(6006))

class TrainModel(Model):
    def __init__(self, learning_rate: float, ultimate_batch_size: int, epochs: int):
        super(TrainModel, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.total_steps = (45_000 // ultimate_batch_size) * epochs
        self.warmup_steps = int(0.1 * self.total_steps)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5, betas=(0.9, 0.98), eps=1e-9)
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
    # lr_moniter = LearningRateMonitor(logging_interval='step')
    # trainer = Trainer(tpu_cores=8, max_epochs=epochs, callbacks=[lr_moniter])
    # datamodule = IMDBDataModule(batch_size=16)
    # model = TrainModel(learning_rate=2e-5, ultimate_batch_size=8*16, epochs=epochs)
    # trainer.fit(model, datamodule)

    pass