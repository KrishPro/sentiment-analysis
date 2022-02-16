"""
Written by KrishPro @ KP
"""

from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from data import PAD_IDX, Dataset
from model import Bert
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch
import time
import os

PWD = ""

LEARNING_RATE = 3e-4
BATCH_SIZE = 32
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = D_MODEL * 2
DROPOUT = 0.1

def load_checkpoint(bert: Bert, checkpoint_path: str = "checkpoints/latest.pth"):
    checkpoint_path = os.path.join(PWD, checkpoint_path)
    if os.path.exists(checkpoint_path):
        bert.load_state_dict(torch.load(checkpoint_path))
    return bert

def save_checkpoint(bert: Bert, checkpoint_path: str = "checkpoints/latest.pth"):
    checkpoint_path = os.path.join(PWD, checkpoint_path)
    torch.save(bert.state_dict(), checkpoint_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataset(BATCH_SIZE)
    dataloader = data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    vocab_size = len(dataset.vocab)

    bert = Bert(D_MODEL, vocab_size, NHEAD, DIM_FEEDFORWARD, NUM_ENCODER_LAYERS, DROPOUT, PAD_IDX)
    bert = load_checkpoint(bert)
    bert = bert.to(device)

    optimizer = optim.Adam(bert.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.BCELoss() 

    writer = SummaryWriter(os.path.join(PWD, f"runs/{time.time()}"))
    global_step = 0

    optimizer.zero_grad()

    with tqdm(dataloader) as pbar:
        for i, (review, target) in enumerate(pbar):

            predictions = bert(review.to(device))

            loss: torch.Tensor = criterion(predictions, target.to(device).float())

            loss.backward()

            writer.add_scalar("loss", loss.item(), global_step=global_step)
            global_step += 1

            if i % 4 == 0:
                pbar.set_postfix(loss=loss.item())
                optimizer.step()
                optimizer.zero_grad()
        
        save_checkpoint(bert)

if __name__ == "__main__":
    main()