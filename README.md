# Sentiment Analysis

![Sentiment Analysis](Assets/sentimen-analysis.jpg)

Sentiment analysis is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information [(Wiki)](https://en.wikipedia.org/wiki/Sentiment_analysis)

## Approaches
- [Multi-Layer LSTM](https://github.com/KrishPro/sentiment-analysis/tree/multi-layer-lstm)
- [Custom Transformer](https://github.com/KrishPro/sentiment-analysis/tree/custom-transformer)
- [Distil Bert](https://github.com/KrishPro/sentiment-analysis/tree/finetuning/distilbert)

For experimental purpose i also tried to copy bert architechure and train it from scratch,
[Bert](https://github.com/KrishPro/sentiment-analysis/tree/fails/bert-scratch)\
But this seems to be failed, I didn't got anytime to debug it

## Top Results
### Custom Transformer
```
Write a review about any movie
=> this movie is very great, i loved this movie 

Classified review as positive, 99.976%
```
```
Write a review about any movie
=> this movie is very bad, i hated the movie

Classified review as negative, 99.691%
```
```
Write a review about any movie
=> the movie was great but actors were worse. the story was cool but the ending wasn't

<=== Almost Neutral ===>
```

### Distil Bert
```
Write a review about any movie
=> this movie is very great, i loved this movie 

Classified review as positive, 99.716%
```
```
Write a review about any movie
=> this movie is very bad, i hated the movie

Classified review as negative, 99.760%
```
```
Write a review about any movie
=> the movie was great but actors were worse. the story was cool but the ending wasn't

Classified review as negative, 85.664%
```
```
Write a review about any movie
=> actors were worse but the movie was great. the ending was bad but the story was awesome

<=== Almost Neutral ===>
```

### Multi-Layer LSTM
They have also produced comparable results, But i made this way before so i don't have any results from it.
Checkout its [branch](https://github.com/KrishPro/sentiment-analysis/tree/multi-layer-lstm) to see its results & loss graph

## Motivation
The main motivation for creating this project was,
I wanted to get hands-on [transformers](https://arxiv.org/pdf/1706.03762.pdf) & [bert finetuning](https://arxiv.org/pdf/1810.04805.pdf)

I am very happy with this project,

## Thank You 🙏️
