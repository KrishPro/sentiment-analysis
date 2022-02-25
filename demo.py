"""
Written by KrishPro @ KP
"""

from transformers.models.distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
from train import TrainModel
import torch


def load_model():
    model: TrainModel = TrainModel(learning_rate=1e-6, ultimate_batch_size=1)
    model.load_state_dict(torch.load("Dist/multi-epoch.ckpt"))
    model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

def take_input():
    print("Write a review about any movie")
    review: str = input("=> ").strip().lower()
    return review

def process_review(review: str, tokenizer: DistilBertTokenizerFast):
    tokens: dict[str, torch.Tensor] = tokenizer(review, return_tensors="pt")
    return tokens

if __name__ == "__main__":
    print("Loading Model...")
    model, tokenizer = load_model()
    print("\n")
    
    while True:
        review = take_input()
        if review == "<exit>":
            break
        tokens = process_review(review, tokenizer)
        logit: torch.Tensor = model(**tokens)

        result = torch.sigmoid(logit).item()

        if not (0.4 < result < 0.6):
            sentiment: str = "positive" if result > 0.5 else "negative"
            confidence: float = result if result > 0.5 else (1 - result)
            print()
            print(f"Classified review as {sentiment}, {(confidence*100):.3f}%")

        else:
            print()
            print("<=== Almost Neutral ===>")

        print("\n")
        print("------=+=+=+=+=------")
        print("\n")