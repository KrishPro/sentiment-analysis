"""
Written by KrishPro @ KP
"""

from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

import pytorch_lightning as pl
import torch.utils.data as data
import pandas as pd
import torch
import os

tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base", use_fast=False)

class Dataset(data.Dataset):
    def __init__(self, data_dir, split="train"):
        self.data = pd.read_csv(os.path.join(data_dir, f"{split}_data.csv"))
        if split == "train":
            self.data = pd.concat([self.data[self.data['sentiment'] == 0].sample(frac=0.25).reset_index(drop=True), self.data[self.data['sentiment'] == 1].sample(frac=0.25).reset_index(drop=True)]).sample(frac=1).reset_index(drop=True)
    def __getitem__(self, idx):
        return tuple(self.data.iloc[idx])
    def __len__(self):
        return len(self.data)

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train = Dataset(self.data_dir, "train")
            self.val = Dataset(self.data_dir, "test")
     
    @staticmethod
    def collate_fn(inp):
        sentences, sentiments = list(zip(*inp))
        return tokenizer(list(sentences), padding="max_length", truncation=True, return_tensors="pt"), torch.tensor(sentiments)

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)