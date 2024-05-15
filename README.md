# Text Classification with Scikit-learn

## Introduction

This project focuses on text classification for the OffensEval shared task, which involves identifying offensive language in social media posts. Specifically, we address SubTasks A and B of the 2019 edition:

- **SubTask A**: Offensiveness Identification (Binary Text Classification)
- **SubTask B**: Offense Type Categorization (Multiclass Text Classification)

We utilize scikit-learn, a Python library for machine learning, to implement and evaluate the classification models.

## Data Loading

The dataset consists of 13240 tweets for training and 860 tweets for testing, with annotations for both SubTasks A and B. The data can be loaded into Pandas DataFrames using the following code:

```python
import pandas as pd

train = pd.read_csv("data/train.tsv", sep="\t")
test = pd.read_csv("data/test.tsv", sep="\t")
```

## Text Representation
To convert the text of tweets into numerical feature vectors, we use a bag-of-words representation based on TF-IDF. The TfidfVectorizer from scikit-learn is employed for this purpose.
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidfvectorizer():
    return TfidfVectorizer()

vectorizer = create_tfidfvectorizer()
train_x, test_x = run_vectorizer(vectorizer, train["tweet"], test["tweet"])
```

## Logistic Regression
We train a Logistic Regression classifier using scikit-learn, with default options and increased maximum iterations for convergence.
```python
from sklearn.linear_model import LogisticRegression

def create_model():
    return LogisticRegression(max_iter=1000)

model = create_model()
```

## Training and Evaluation
We train the model on the training data and evaluate its performance on the test set.
```python
prediction_a = run_model(model, train_x, train["subtask_a"], test_x)
prediction_b = run_model(model, train_x, train["subtask_b"], test_x)
```

## Balancing the Dataset
To handle class imbalance in the dataset, we create a balanced version of the Logistic Regression model by adjusting class weights.
```python
def create_balanced_model():
    return LogisticRegression(class_weight='balanced', max_iter=1000)

balanced_model = create_balanced_model()
```

## Additional Features
Incorporating additional features, such as sentiment analysis results, can enhance the classification performance. We utilize a ColumnTransformer to combine text and sentiment features.
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def create_column_transformer():
    transformers = [
        ('text', TfidfVectorizer(), 'tweet'),
        ('sentiment', OneHotEncoder(), ['sentiment'])
    ]
    return ColumnTransformer(transformers, remainder='drop')

column_transformer = create_column_transformer()
train_x_sentiment, test_x_sentiment = run_vectorizer(column_transformer, train, test)
```

## Conclusion
This project demonstrates the application of text classification techniques using scikit-learn for the OffensEval task. By leveraging logistic regression and incorporating additional features like sentiment analysis, we aim to improve the accuracy and effectiveness of offensive language detection in social media posts.
