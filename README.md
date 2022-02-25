# Sentiment Analysis (Fine-Tuning DistilBert)

![Sentiment Analysis (BERT from Scratch)](Assets/bert.png)

## Architecture
In terms of architecture, I haven't done anything special.

I have just used plain pre-trained distil bert and passed it's class label to my classifier

**My Classifier**
```python
Sequential(
  (0): LeakyReLU(negative_slope=0.01)
  (1): Dropout(p=0.5, inplace=False)
  (2): Linear(in_features=768, out_features=192, bias=True)
  (3): LeakyReLU(negative_slope=0.01)
  (4): Dropout(p=0.5, inplace=False)
  (5): Linear(in_features=192, out_features=1, bias=True)
)
```
Note: I haven't used sigmoid inside my model as I'm using `BCEWithLogitsLoss` as my loss function

## Results
I don't remember exactly (I've also lost my tensorboard logs).
But as i do, We got an accuracy on test set of about `90%-95%`

This time the weight file was even larger than 100mb (about 266mb), So that's why i haven't uploaded the weight for this branch

But, Google Drive comes to save life. Get the checkpoints from [here](https://drive.google.com/file/d/17K6r_0wJ5GsCXWn_ZVOV6J6QSMnyqgHv/view?usp=sharing)

## Demo
Note: Some reviews are copied from [other branch](https://github.com/KrishPro/sentiment-analysis/tree/custom-transformer) for proper comparision\
**Below are some predictions made by my model**

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
=> ----

<=== Almost Neutral ===>
```

