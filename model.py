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

        self.dim: int = self.config.dim
        self.dropout: float = 0.5

        self.classifier = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 1),
            nn.Sigmoid()
        )

        self.reset_parameters()
        
    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        
        bert_output: BaseModelOutput = self.model(input_ids, attention_mask)
       
        cls_token = bert_output.last_hidden_state[:, 0, :]

        assert (cls_token.dim() == 2) and (cls_token.size(1) == self.dim), f"cls_token shape must be ({input_ids.size(0)}, {self.dim}), Got {cls_token.shape}"

        output: torch.Tensor = self.classifier(cls_token)

        return output.squeeze(1)
