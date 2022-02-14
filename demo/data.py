"""
Written by KrishPro @ KP
"""

from torchtext.data.utils import get_tokenizer

import re
from torchtext.vocab import vocab

import torch

def clean_text(text):
    text = text.lower()
    text = text.replace("<br />", "")
    text = re.sub(r'[^\w\s]', "", text)
    return text

def build_vocab():
    word_count = torch.load("static/vocab.pth")

    voc = vocab(word_count, min_freq=2)
    special_tokens = ('<PAD>', '<UNK>')
    for i, tok in enumerate(special_tokens): voc.insert_token(tok, i)
    voc.set_default_index(special_tokens.index("<UNK>"))
    
    return voc
