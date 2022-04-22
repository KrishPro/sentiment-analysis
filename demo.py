"""
Written by KrishPro @ KP
"""

from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from utils import process_text
from model import Model

import torch


# the checkpoint file is stored online here:
# https://drive.google.com/file/d/1sHhIGvexAEryW9PRPACOnFvQUjyw5BWD/view?usp=sharing
def load_model(model_name, checkpoint_path="Others/latest.ckpt"):
    state_dict = torch.load(checkpoint_path)['state_dict']
    model = Model(model_name)
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def main():
    print("Loading model & tokenizer...")
    model_name="distilroberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = load_model(model_name)
    while True:
        print()
        print('='*10, 'Quick Demo', '='*20)
        text = process_text(input("$ "))
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        result: float = model(inputs).detach().sigmoid().item()
        
        if not (0.4 < result < 0.6):
            sentiment: str = "positive" if result > 0.5 else "negative"
            confidence: float = result if result > 0.5 else (1 - result)
            print()
            print(f"Classified review as {sentiment}, {(confidence*100):.3f}%")

        else:
            print()
            print("<=== Almost Neutral ===>")

if __name__ == "__main__":
    main()