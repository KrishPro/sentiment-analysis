from flask import Flask
from data import get_tokenizer, build_vocab, clean_text
from flask.templating import render_template
from model import Model
from hyperparams import *
import torch

app = Flask(__name__)

device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu") 
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
vocab = build_vocab()

net = Model(HIDDEN_SIZE, EMBED_SIZE, len(vocab), N_LAYERS).to(device)
state = torch.load("static/model.pth")
net.load_state_dict(state)

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/analyse/<text>")
def analyse(text):
    with torch.no_grad():
        review = torch.tensor(vocab(tokenizer(clean_text(text))), device=device).view(1, -1)
        y = net(review).item()
        
    return render_template("results.html", positive = y*100, negative = (1 - y)*100, text=text)

app.run(debug=True)