#!/usr/bin/env python3

import string

import nltk


#TODO this doesn't work yet, the CSV has tags and sentences. We need to split them up
# and then do the preprocessing on the sentences to return a list of lists of words, not letters
def load_and_preprocess_data(path: str, remove_punct: bool, lowercase: bool) -> "list[list[str]]":
    """
    Loads and preprocesses the data from the NLTK Treebank corpus.
    Args:
        None
    Returns:
        List[List[str]]: A list of lists, where each inner list represents a sentence
            from the Treebank corpus after removing punctuation, converting to lowercase,
            and removing empty tokens.
    """
    # Remove punctuation and lowercase, removing empty tokens after cleaning

    with open(path, 'r', encoding="ISO-8859-1") as file_in:
        data = file_in.readlines()
    corpus = list()
    for sent in data:
        if remove_punct:
            sent = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent]
        if lowercase:
            sent = [word.lower() for word in sent]
        corpus.append(" ".join(sent).split())
    return corpus

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