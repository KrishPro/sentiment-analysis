"""
Written by KrishPro @ KP
"""

import random
from torchtext.vocab import vocab as build_vocab, Vocab
from torchtext.data.utils import get_tokenizer
from train import BATCH_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
from torch.utils import data
from model import Transformer
from data import PAD_IDX, Dataset
from tqdm import tqdm
import torch.nn as nn
import torch
import os

def load_vocab(vocab_path: str = "Data/word-count.pth", min_freq: int = 2) -> Vocab:
    counter = torch.load(vocab_path)

    vocab = build_vocab(counter, min_freq)
    vocab.insert_token("[UNK]", len(vocab))
    vocab.insert_token("[PAD]", PAD_IDX)
    vocab.set_default_index(vocab.get_stoi()["[UNK]"])

    return vocab

def take_input(tokenizer, vocab, device=None):
    print("Write a review about any movie")
    review = input("=> ").lower().strip()
    review: list[str] = tokenizer(review)
    review: list[int] = vocab(review)
    review: torch.Tensor = torch.tensor(review, dtype=torch.long, device=device).unsqueeze(1)
    assert (len(review.shape) == 2) and review.size(1) == 1, f"review shape should be (-1, 1), Got {tuple(review.shape)}"
    return review

@torch.no_grad()
def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    dataset = Dataset(128)
    random.shuffle(dataset.data)
    dataset.data = dataset.data[:50]

    vocab = load_vocab()
    vocab_size = len(vocab)

    transformer = Transformer(D_MODEL, vocab_size, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT, PAD_IDX)
    transformer = transformer.to(device)
    transformer.load_state_dict(torch.load("Dist/final-version.pth"))
    transformer.eval()

    criterion = nn.BCELoss()

    loss = 0

    for review, target in tqdm(dataset):

        predictions = transformer(review.to(device))

        loss += criterion(predictions, target.to(device).float())

         
    print(loss.item() / len(dataset))

    

if __name__ == '__main__':
    main()
    