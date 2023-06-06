#!/usr/bin/env python3

import string

import nltk
import pandas as pd
import sklearn.feature_extraction.text as sklfe
import sklearn.metrics
from numpy import ndarray


def confusion_matrix(y_true: "list[int]", y_pred: "list[int]|ndarray"): # -> "ndarray":
    """
    Create a confusion matrix from two lists of labels.
    Args:
        y_true (list[int]): The true labels.
        y_pred (list[int]): The predicted labels.
    Returns:
        ndarray: The confusion matrix.
    """
    return pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    # return sklearn.metrics.confusion_matrix(y_true, y_pred)


def load_and_preprocess_data(path: str, remove_punct: bool, lowercase: bool) -> "list[list[str]]":
    """#TODO """

    dataframe = pd.read_csv(path, encoding="ISO-8859-1", header=None)
    data = dataframe[1].tolist()
    corpus = list()
    for sent in data:
        sent = sent.split()
        if remove_punct:
            sent = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent]
        if lowercase:
            sent = [word.lower() for word in sent]
        corpus.append(" ".join(" ".join(sent).split()))
    dataframe[1] = corpus
    return dataframe

def vectorize_set(corpus: "list[list[str]]") -> "tuple[ndarray[spmatrix]": # type:ignore
    # encode sentences as one-hot vectors using scikit-learn's one_hot function OHE
    vectorizer = sklfe.CountVectorizer()
    return vectorizer.fit_transform(corpus).toarray() # type: ignore
 
def test(confusion_matrix: "pd.DataFrame", classifier: str) -> None:

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

    if classifier == 'XGBoost':
        TP = confusion_matrix[2][2]
        TN = confusion_matrix[0][0]
        FP = confusion_matrix[0][2] + confusion_matrix[1][2]
        FN = confusion_matrix[1][0] + confusion_matrix[2][0]
    else: # classifier == 'Naive Bayes':
        TP = confusion_matrix['positive']['positive']
        TN = confusion_matrix['negative']['negative']
        FP = confusion_matrix['negative']['positive'] + confusion_matrix['neutral']['positive']
        FN = confusion_matrix['neutral']['negative'] + confusion_matrix['positive']['negative']

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"\n{classifier} accuracy:", accuracy)
    precision = TP / (TP + FP)
    print(f"{classifier} precision:", precision)
    recall = TP / (TP + FN)
    print(f"{classifier} recall:", recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"{classifier} F1:", f1)
    return None

def train_test_split(corpus: "list[list[str]]", train_ratio: float = 0.8) -> "tuple[list[list[str]], list[list[str]]]":
    """
    Split the corpus into train and test sets using a 80:20 ratio..
    Args:
        corpus (list[list[str]]): The input corpus to be split.
        train_ratio (float): The ratio of the train set to the test set.
    Returns:
        train: (list[list[str]]), test: (list[list[str]]) : A tuple containing the train and test set splits.
    """
    return corpus[:int(len(corpus) * train_ratio)], corpus[int(len(corpus) * train_ratio):]