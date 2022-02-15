"""
Written by KrishPro @ KP
"""

import torch.nn as nn
import torch
import math

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
    def __init__(self, vocab_size: int, emb_size: int, pad_idx: int, dropout: float):
        super(EmbeddingLayer, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.emb_size = emb_size

    def forward(self, indicies: torch.Tensor):
        assert indicies.dtype == torch.long, f"indicies given to embedding layer must be of dtype long, Got {indicies.dtype}"
        
        embeddings: torch.Tensor = self.embedding_layer(indicies) * math.sqrt(self.emb_size)
        return self.positional_encoding(embeddings)
