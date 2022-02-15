"""
Written by KrishPro @ KP
"""

from custom_decoder import TransformerDecoderLayer
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

class Transformer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, nhead: int, num_encoder_layers: int,
                num_decoder_layers: int, dim_feedforward: int, dropout: float, pad_idx: int
                ):
        super(Transformer, self).__init__()
            
        self.pad_idx = pad_idx
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model, self.pad_idx, dropout)

        self.cls_embed = nn.Embedding(1, d_model)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, custom_decoder=decoder)

        self.classifier = nn.Linear(d_model, 1)

    def create_pad_mask(self, tensor: torch.Tensor):
        return (tensor == self.pad_idx).T

    def get_cls_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.cls_embed(torch.tensor([0], device=device).long().expand(1, batch_size))

    def forward(self, x: torch.Tensor):
        # x.shape: (S, N)

        # Creating Pad Masks
        pad_mask: torch.Tensor = self.create_pad_mask(x)
        memory_pad_mask = pad_mask.clone()
        assert pad_mask.device == memory_pad_mask.device == x.device, "All masks should be on the same device"

        # Embedding the review
        x = self.embedding_layer(x)

        # Creating [CLS] token
        cls_token = self.get_cls_token(x.size(1), device=x.device)

        # Processing the embedding review
        out: torch.Tensor = self.transformer(x, cls_token, src_key_padding_mask=pad_mask, memory_key_padding_mask=memory_pad_mask)

        # Using the output of transformer to classifiy the review
        out = torch.sigmoid(self.classifier(out.squeeze(0)))

        return out # out.shape: (N, 1)
        