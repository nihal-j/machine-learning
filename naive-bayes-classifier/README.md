# Naive Bayes Classifier

The objective of this project is to implement a Naive Bayes classifier model from scratch for sentiment analysis.
Each piece of text (a product review) is to be classified as a positive review or a negative review.
Additionally, to evaluate the model, 5-fold cross validation is used.

#### Methodology

- Preprocessing done: Punctuation symbols were removed.
- The data was shuffled randomly before building the model.
- In general, any string (while training or testing), is transformed to a binary incidence vector in the vocabulary space.
- Laplace smoothing is done when estimating the likelihood values.
- In a test, unknown words are discarded

#### Sample tests

Training samples can be found in `data/`. Following examples are self-generated.
Following predictions are done on a model that is tested on all the given examples.

```
Comment: This product is amazing.
Sentiment: 1
Predicted: 1

Comment: I don't like it at all!
Sentiment: 0
Predicted: 0
```

#### Metrics Used

The following metrics are calculated for each cross-validation set:
- **Accuracy** is calculated as the percentage of correctly classified samples.
- **F-score** is calculated as the harmonic mean of precision and recall, where:
-- precision is calculated as ratio of true positives and predicted positives
-- recall is calculated as ratio of true positives and actual positives

#### Results

- Using bag of words model (without Laplace smoothing)

|Name|Accuracy|F-score|
|--|--|--|
Model 0| 75.0|0.77
Model 1| 70.0|0.71
Model 2| 72.0|0.75
Model 3| 72.5|0.76
Model 4| 76.0|0.79

Average validation accuracy:  **73.1 +/- 2.15**

---

- Using bag of words model (with Laplace smoothing)

|Name|Accuracy|F-score|
|--|--|--|
Model 0| 79.5|0.8
Model 1| 80.0|0.798
Model 2| 87.5|0.88
Model 3| 76.5|0.779
Model 4| 84.5|0.85

Average validation accuracy:  **81.6 +/- 3.90**

---

- Using binary vectors (with Laplace smoothing)

|Name|Accuracy|F-score|
|--|--|--|
Model 0| 78.0|0.77
Model 1| 81.5|0.798
Model 2| 86.0|0.858
Model 3| 77.5|0.78
Model 4| 86.5|0.87

Average validation accuracy:  **81.9 +/- 3.81**

---

#### Conclusions

- The models perform well on unseen data but overall the models have high variability as indicated by the standard deviation of the accuracy.
- Using binary vectors does slightly better than using bag of words model
- Use of Laplace smoothing provides significant gains in accuracy.