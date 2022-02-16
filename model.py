"""
Written by KrishPro @ KP
"""

from torch.nn import functional as F
import torch.nn as nn
import torch
import math

# `PositionalEncoding` is copied from https://pytorch.org/tutorials/beginner/translation_transformer.html#seq2seq-network-using-transformer
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, dropout: float, padding_idx: int):
        super(EmbeddingLayer, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.emb_size = emb_size

    def forward(self, indices: torch.Tensor):
        assert indices.dtype == torch.long, f"indices.dtype must be torch.long, Got {indices.dtype}"
        
        embeddings: torch.Tensor = self.embedding_layer(indices) * math.sqrt(self.emb_size)
        embeddings: torch.Tensor = self.positional_encoding(embeddings)
        return embeddings
