"""
Written by KrishPro @ KP
"""

from torchtext.vocab import vocab as build_vocab, Vocab
from torchtext.data.utils import get_tokenizer
from train import D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
from model import Transformer
from data import PAD_IDX
import torch

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    vocab = load_vocab()
    vocab_size = len(vocab)

    transformer = Transformer(D_MODEL, vocab_size, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT, PAD_IDX)
    transformer = transformer.to(device)
    transformer.load_state_dict(torch.load("Dist/final-version.pth"))
    transformer.eval()

    review = take_input(tokenizer, vocab, device)
    result = transformer(review).item()

    if not (0.4 < result < 0.6):
        sentiment: str = "positive" if result > 0.5 else "negative"
        confidence: float = result if result > 0.5 else (1 - result)
        print()
        print(f"Classified review as {sentiment}, {(confidence*100):.3f}%")

    else:
        print()
        print("<=== Almost Neutral ===>")
        
    

if __name__ == '__main__':
    main()
    