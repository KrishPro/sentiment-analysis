"""
Written by KrishPro @ KP
"""

from transformers import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import DistilBertModel, BaseModelOutput
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch

class Model(LightningModule):
    def __init__(self):
        super(Model, self).__init__()

        self.config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.size: int = self.config.dim // 4
        self.dropout: float = 0.5

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.size, self.size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.size, self.size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.size, 1),
        )

        self.reset_parameters()
        
    def reset_parameters(self):
        for param in self.classifier.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        
        bert_output: BaseModelOutput = self.model(input_ids, attention_mask)
       
        cls_token = bert_output.last_hidden_state[:, 0, :]

        output: torch.Tensor = self.classifier(cls_token)

        return output.squeeze(1)
