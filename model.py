"""
Written by KrishPro @ KP
"""

from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig

import torch.nn.functional as F
import torch.nn as nn
import torch

class ClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, activation=F.leaky_relu, dropout_prob=0.5):
        super(ClassifierHead, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, 1)
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, cls_token: torch.Tensor):
        hidden_state: torch.Tensor = self.dropout(cls_token)
        hidden_state: torch.Tensor = self.activation(hidden_state)
        
        hidden_state: torch.Tensor = self.linear1(hidden_state)
        hidden_state: torch.Tensor = self.activation(hidden_state)
        hidden_state: torch.Tensor = self.dropout(hidden_state)
        
        hidden_state: torch.Tensor = self.linear2(hidden_state)
        return hidden_state

class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.config = RobertaConfig.from_pretrained(model_name)
        self.classifier = ClassifierHead(self.config.hidden_size)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs: dict[str, torch.Tensor]):
        (encoded_words, _) = self.roberta(**inputs, return_dict=False)
        cls_token: torch.Tensor = encoded_words[:, 0, :] # (batch, time, hidden_size)
        output: torch.Tensor = self.classifier(cls_token)
        return output.squeeze(1)
