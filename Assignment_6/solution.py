#!/usr/bin/env python3

import string

import nltk
import pandas as pd


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