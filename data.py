"""
Written by KrishPro @ KP
"""

from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, vocab as build_vocab
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from tqdm import tqdm
import pandas as pd
import torch
import os
import re

PAD_IDX = 0

class Dataset(data.Dataset):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        
        tqdm.pandas()

        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

        df: pd.DataFrame = self.load_data(self.tokenizer)

        self.vocab = self.create_vocab(df['text'], min_freq=2)
        
        df['text'] = df['text'].apply(self.vocab)

        df = df[df['text'].map(len) < 500]

        df['text_len'] = df['text'].map(len)

        df = df.sort_values('text_len')
        df = df.drop('text_len', axis=1).reset_index().drop('index', axis=1)
        
        self.data = df.values.tolist()
        self.data = list(self.chunks(self.data, batch_size))[:-1]

    def __getitem__(self, idx):
        rev, label = self.data[idx]
        rev = [torch.tensor(r) for r in rev]
        label = torch.tensor(label)
        return pad_sequence(rev, padding_value=PAD_IDX), label
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield [l[0] for l in lst[i:i + n]], [l[1] for l in lst[i:i + n]]

    def load_data(self, data_dir: str = "/home/krish/Datasets/IMDB-Pos-vs-Neg"):
        if os.path.exists("Data/data.json"):
            return pd.read_json("Data/data.json")

        else:
            splits: list[str] = os.listdir(data_dir)

            file_paths = [os.path.join(data_dir, file_name) for file_name in splits]
            dataframe = pd.concat(list(map(pd.read_csv, file_paths)), ignore_index=True)
            dataframe['text'] = dataframe['text'].map(self.clean_text)
            dataframe['text'] = dataframe['text'].progress_map(self.tokenizer)
            return dataframe

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = text.replace("<br />", "")
        text = re.sub(r'[^\w\s]', "", text)
        return text

    def create_vocab(self, text: pd.Series, min_freq: int) -> Vocab:
        if os.path.exists("Data/word-count.pth"):
            counter = torch.load("Data/word-count.pth")
        else:
            counter = Counter()
            for t in tqdm(text):
                counter.update(t)
            torch.save(counter, "Data/word-count.pth")


        vocab = build_vocab(counter, min_freq)
        vocab.insert_token("[UNK]", len(vocab))
        vocab.insert_token("[PAD]", PAD_IDX)
        vocab.set_default_index(vocab.get_stoi()["[UNK]"])

        return vocab

def main():
    dataset = Dataset(64)
    dataloader = data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    rev, tar = next(iter(dataloader))

    print(rev.shape, tar.shape)

if __name__ == '__main__':
    main()