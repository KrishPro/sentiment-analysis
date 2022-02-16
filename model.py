"""
Written by KrishPro @ KP
"""

from typing import Callable, Union
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

class Bert(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, nhead: int, dim_feedforward: int, num_encoder_layers: int, dropout: float, padding_idx: int, activation:  Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):
        super(Bert, self).__init__()

        self.pad_idx = padding_idx
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, dropout, self.pad_idx)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.classifier = nn.Linear(d_model, 1)

    def create_pad_mask(self, r: torch.Tensor):
        return (r == self.pad_idx).T

    def forward(self, r: torch.Tensor):
        # r.shape: (S, N)
        padding_mask: torch.Tensor = self.create_pad_mask(r)
        r: torch.Tensor = self.embedding_layer(r)
        # r.shape: (S, N, E)
        mem: torch.Tensor = self.encoder(r, mask=None, src_key_padding_mask=padding_mask)
        # mem.shape: (S, N, E)
        mem = mem[0] # Taking the encoding for the [CLS] token
        # mem.shape: (N, E)
        output: torch.Tensor = self.classifier(mem)
        # output.shape: (N, 1)
        return torch.sigmoid(output).squeeze(1)

def main():
    bert = Bert(d_model=512, vocab_size=30_000, nhead=8, dim_feedforward=512, num_encoder_layers=6, dropout=0.1, padding_idx=0)

    fake_reviews = torch.randint(0, 29_999, (150, 64))
    fake_outs: torch.Tensor = bert(fake_reviews)

    print(fake_reviews.shape, fake_outs.shape)


if __name__ == '__main__':
    main()