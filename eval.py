"""
Written by KrishPro @ KP
"""

from torchmetrics import Accuracy
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from utils import process_text
from model import Model
import pandas as pd

import torch.nn.functional as F
import torch


# the checkpoint file is stored online here:
# https://drive.google.com/file/d/1sHhIGvexAEryW9PRPACOnFvQUjyw5BWD/view?usp=sharing
def load_model(model_name, checkpoint_path="Others/latest.ckpt"):
    state_dict = torch.load(checkpoint_path)['state_dict']
    model = Model(model_name)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_accuracy(results):
    r = list(map(lambda s: s[1] == s[2], results))
    return (sum(r) / len(r)) * 100

def get_bce(results):
    r = torch.tensor(list(map(lambda s: (float(s[0]), float(s[2])), results)))
    # r.shape = (num_samples, 2)
    preds = r.T[0]
    trues = r.T[1]
    loss = F.binary_cross_entropy(preds, trues)
    return loss.item()


@torch.no_grad()
def main():
    print("Loading model & tokenizer...")
    model_name="distilroberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = load_model(model_name)
    test_data = pd.read_csv('Data/test_data.csv')
    results = []
    for _, text, target in test_data.itertuples():
       
        text = process_text(text)
      
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        prob: float = model(inputs).detach().sigmoid().item()
        pred = int(prob > 0.5)
        results.append((prob, pred, target))

    loss = get_bce(results)
    accuracy = get_accuracy(results)

    print()
    print(f"Loss     = {loss:.3f}")
    print(f"Accuracy = {accuracy:.3f}%")
        

if __name__ == "__main__":
    main()