"""
Written by KrishPro @ KP
"""

from typing import Optional
from pytorch_lightning import LightningDataModule

import torch.utils.data as data
import pandas as pd
import shutil
import torch
import kaggle
import os

from transformers.models.distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast


class Dataset(data.Dataset):
    def __init__(self, split: str = "Train"):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        csv: pd.DataFrame = pd.read_csv(f"Data/{split}.csv")
        self.data = csv.to_numpy()

    def __getitem__(self, idx):
        review, label = tuple(self.data[idx])
        label = torch.tensor(label, dtype=torch.float)

        tokens: tuple[torch.Tensor, torch.Tensor] = tuple(self.tokenizer(review, padding="max_length", truncation=True, return_tensors="pt").values())
        input_ids, attention_mask =  tokens

        return (input_ids.squeeze(0), attention_mask.squeeze(0)), label

    def __len__(self):
        return len(self.data)


def download_and_unzip_data(dataset_name: str, output_dir: str):
    """
    But I just made it because I challenged myself
    """

    # This goes inside if-block if, output_dir doesn't exist or if it exists it is empty

    if (not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0):
 
        try: shutil.rmtree(output_dir)
        except FileNotFoundError: pass

        kaggle.api.authenticate()

        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)


class IMDBDataModule(LightningDataModule):
    def __init__(self, batch_size = 16):
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        """
        Dataset is available at Kaggle
        https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format
        """

        download_and_unzip_data("columbine/imdb-dataset-sentiment-analysis-in-csv-format", output_dir = "Data/")

    def setup(self, stage: Optional[str] = None):
        if (stage == "fit") or (stage == None):
            self.train_dataset = Dataset("Train")
            self.val_dataset = Dataset("Valid")

        if (stage == "test") or (stage == None):
            self.test_dataset = Dataset("Test")

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size)
