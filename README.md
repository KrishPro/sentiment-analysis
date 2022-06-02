# Sentiment Analysis
> bert-base-uncased

## Conclusion
`TPU v3-8` is not capable of finetuning `bert-base`.
So, Instead of finetuning whole `bert-base`, I indeed used the `bert-base` as it is and just trainned a classifier to classify cls_token into `positive-vs-negative`.

**How can i say this will work ?**\
Transfer learning in computer vision works like this only.

