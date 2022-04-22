"""
Written by KrishPro @ KP
"""

from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from model import ClassifierHead

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

class Model(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-5):
        super(Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.config = RobertaConfig.from_pretrained(model_name)
        self.classifier = ClassifierHead(self.config.hidden_size)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = learning_rate
        
    def forward(self, inputs):
        (encoded_words, _) = self.roberta(**inputs, return_dict=False)
        cls_token = encoded_words[:, 0, :] # (batch, time, hidden_size)
        output = self.classifier(cls_token)
        return output.squeeze(1)
    
    def training_step(self, batch, batch_idx):
        sentences, targets = batch
        inputs = self(sentences)
        loss = self.criterion(inputs, targets.float())
        print(f"EPOCH {self.current_epoch}| {batch_idx}/23836 ({(batch_idx/23836)*100}) | {loss:.5f}")
        self.log("loss", loss.item())
        return loss
    
    def validation_step(self, batch: tuple, batch_idx: int):
        sentences, targets = batch
        inputs = self(sentences)
        loss = self.criterion(inputs, targets.float())
        self.log("val_loss", loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        roberta = {
            'decay': list(map(lambda s: s[1], filter(lambda s: s[0].endswith('weight'), self.roberta.named_parameters()))),
            'no_decay': list(map(lambda s: s[1], filter(lambda s: s[0].endswith('bias'), self.roberta.named_parameters()))),
            'weight_decay': 0.02,
            'learning_rate': 1e-5
        }
        
        classifier = {
            'decay': list(map(lambda s: s[1], filter(lambda s: s[0].endswith('weight'), self.classifier.named_parameters()))),
            'no_decay': list(map(lambda s: s[1], filter(lambda s: s[0].endswith('bias'), self.classifier.named_parameters()))),
            'weight_decay': 0.02,
            'learning_rate': 3e-4
        }
        
        
        params = [
            {'params': roberta['decay'], 'lr': roberta['learning_rate'], 'weight_decay': roberta['weight_decay']},
            {'params': roberta['no_decay'], 'lr': roberta['learning_rate'], 'weight_decay': 0.0},
            
            {'params': classifier['decay'], 'lr': classifier['learning_rate'], 'weight_decay': classifier['weight_decay']},
            {'params': classifier['no_decay'], 'lr': classifier['learning_rate'], 'weight_decay': 0.0}
        ]
        return optim.Adam(params, lr=self.lr)