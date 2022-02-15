"""
Written by KrishPro @ KP
"""

import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch
import os

from data import PAD_IDX, Dataset
from model import Transformer

LEARNING_RATE = 3e-4
BATCH_SIZE = 4
D_MODEL = 256
NHEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 1
DIM_FEEDFORWARD = D_MODEL
DROPOUT = 0.1

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset(BATCH_SIZE)
    dataloader = data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    vocab_size = len(dataset.vocab)

    transformer = Transformer(D_MODEL, vocab_size, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT, PAD_IDX)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    optimizer = optim.Adam(transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.BCELoss()

    writer = SummaryWriter(f"runs/{time.time()}")
    global_step = 0

    optimizer.zero_grad()

    with tqdm(dataloader) as pbar:
        for i, (review, target) in enumerate(pbar):

            predictions = transformer(review.to(device))

            loss: torch.Tensor = criterion(predictions, target.to(device).float())

            loss.backward()

            writer.add_scalar("loss", loss.item(), global_step=global_step)
            global_step += 1

            if i % 4 == 0:
                pbar.set_postfix(loss=loss.item())
                optimizer.step()
                optimizer.zero_grad()
            
            
        
        

if __name__ == '__main__':
    main()
                