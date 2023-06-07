import string

import nltk
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sklfe
from nltk import ngrams
from nltk.corpus import stopwords
from numpy import ndarray
from sklearn.model_selection import train_test_split

nltk.download('stopwords')


def confusion_matrix(y_true: "list[int]|ndarray", y_pred: "list[int]|ndarray"): # -> "ndarray":
    """
    Create a confusion matrix from two lists of labels.
    Args:
        y_true (list[int]): The true labels.
        y_pred (list[int]): The predicted labels.
    Returns:
        ndarray: The confusion matrix.
    """
    sentiments: dict = {"positive": "2", "neutral": "1", "negative": "0"}
    reversed_dict = {sentiments[k]:k for k in sentiments}
    y_pred = np.array([reversed_dict[str(elem)] for elem in y_pred])
    y_true = np.array([reversed_dict[str(elem)] for elem in y_true])
    # y_true = y_true.map(sentiments)
    # y_pred = y_pred.map(sentiments)
    return pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

def load_and_preprocess_data(path: str="./all-data.csv", ngramize: int=0, polarize: bool=False,
                            remove_stops: bool=False) -> "list[list[str]]":
    """
    Load and preprocess data from a CSV file.
    Args:
        path (str, optional): The path to the CSV file. Defaults to "./all-data.csv".
        ngramize (int, optional): The value specifying the degree of n-gramization. Defaults to 0.
        polarize (bool, optional): Flag indicating whether to polarize the corpus. Defaults to False.
        remove_stops (bool, optional): Flag indicating whether to remove stopwords from the corpus. Defaults to False.
    Returns:
        List[List[str]]: A list of preprocessed sentences, where each sentence is represented as a list of words.
    """
    dataframe = pd.read_csv(path, encoding="ISO-8859-1", header=None)
    data = dataframe[1].tolist()
    corpus = list()
    for sent in data:
        sent = sent.split()
        sent = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent]
        sent = [word.lower() for word in sent]
        corpus.append(" ".join(" ".join(sent).split()))
    if remove_stops:
        corpus = remove_stopwords(corpus)   
    if polarize:
        corpus = polarize_corpus(corpus, load_loughlan_dict(path))
    if ngramize != 0:
        corpus = ngramize_corpus(corpus, ngramize)
    dataframe[0] = replace_labels_with_numbers(dataframe[0])
    dataframe[1] = corpus
    return dataframe


def load_loughlan_dict(path: str) -> "dict[str, dict[str, str]]":
    """
    Load the Loughran-McDonald sentiment dictionary from the given CSV file.

    Args:
        path (str): The path to the CSV file containing the sentiment dictionary.

    Returns:
        Dict[str, Dict[str, str]]: A nested dictionary representing the sentiment dictionary,
        where each word is mapped to its corresponding sentiment polarity.
    """
    sem_df = pd.read_csv(path)
    sem_df["Word"] = sem_df["Word"].str.lower()
    sem_dict = sem_df.set_index("Word").to_dict("index")
    return sem_dict


def ngramize_corpus(corpus: "list[list[str]]|list[list[int]]", n: int = 2) -> "list[list[tuple[str]]]":
    """
    Generate n-grams from the given corpus.
    Args:
        corpus (List[List[str]]): The corpus containing preprocessed sentences.
        n (int, optional): The degree of n-gramization. Defaults to 2.
    Returns:
        List[List[Tuple[str]]]: A list of n-grams, where each n-gram is represented as a tuple of words.
    """
    corpus_new = []
    for sent in corpus:
        sent_new = list(ngrams(sent, n=n))
        corpus_new.append(sent_new)
    return corpus_new


def polarize_corpus(corpus: "list[list[str]]", sem_dict) -> "list[list[int]]":
    """
    Polarize the given corpus using a sentiment dictionary.

    Args:
        corpus (List[List[str]]): The corpus containing preprocessed sentences.
        sem_dict (Dict[str, Dict[str, str]]): A sentiment dictionary mapping words to their polarities.

    Returns:
        List[List[int]]: A list of polarized sentences, where each word is represented as an integer polarity value.
    """
    corpus_new = []
    for sent in corpus:
        sent_new = []
        for w in sent:
            try:
                word_pol = sem_dict[w]
                if word_pol["Positive"] != "0":
                    sent_new.append(2)
                elif word_pol["Negative"] != "0":
                    sent_new.append(0)
                else:
                    sent_new.append(1)
            except KeyError:
                sent_new.append(1)
        corpus_new.append(sent_new)
    return corpus_new


def remove_stopwords(corpus: "list[list[str]]") -> "list[list[str]]":
    """
    Remove stopwords from the given corpus.

    Args:
        corpus (List[List[str]]): The corpus containing preprocessed sentences.

    Returns:
        List[List[str]]: A list of sentences with stopwords removed.
    """
    stopWords = set(stopwords.words('english'))
    corpus_new = []
    for sent in corpus:
        sent_new = []
        for w in sent:
            if w not in stopWords:
                sent_new.append(w)
        corpus_new.append(sent_new) 
    return corpus_new


def replace_labels_with_numbers(dataframe: "list[str]") -> "list[str]":
    """
    Replace labels in the corpus with numerical representations.
    Args:
        corpus (List[str]): The corpus containing labeled data.
    Returns:
        List[str]: The corpus with labels replaced by their corresponding numerical representations.
    """
    sentiments: dict = {"positive": 2, "neutral": 1, "negative": 0}
    dataframe = [sentiments[str(elem)] for elem in dataframe]
    return dataframe

def test(confusion_matrix: "pd.DataFrame", classifier: str, preprocessing: str) -> None:
    """
    Calculate and print evaluation metrics based on the provided confusion matrix.
    Args:
        confusion_matrix (pd.DataFrame): A pandas DataFrame representing the confusion matrix.
        classifier (str): The name of the classifier used.
        preprocessing (str): The name of the preprocessing technique used.
    Returns:
        None
    """
    TP = confusion_matrix['positive']['positive']
    TNeutral = confusion_matrix['neutral']['neutral']
    TNegative = confusion_matrix['negative']['negative']
    FP = confusion_matrix['negative']['positive'] + confusion_matrix['neutral']['positive']
    FN = confusion_matrix['neutral']['negative'] + confusion_matrix['positive']['negative']
    Total = (confusion_matrix['negative']['negative'] + confusion_matrix['negative']['neutral'] + confusion_matrix['negative']['positive'] +
                confusion_matrix['neutral']['negative'] + confusion_matrix['neutral']['neutral'] + confusion_matrix['neutral']['positive'] +
                confusion_matrix['positive']['negative'] + confusion_matrix['positive']['neutral'] + confusion_matrix['positive']['positive'])

    # Accuracy = (TP + TNeut + TNeg) / (Total)
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

    accuracy = (TP + TNeutral + TNegative) / Total
    print(f"\n{preprocessing} {classifier} accuracy:", accuracy)
    precision = TP / (TP + FP)
    print(f"{preprocessing} {classifier} precision:", precision)
    recall = TP / (TP + FN)
    print(f"{preprocessing} {classifier} recall:", recall)
    F1 = 2 * (precision * recall) / (precision + recall)
    print(f"{preprocessing} {classifier} F1:", F1)
    return None

def train_and_fit_model(corpus: "tuple[list[str|int], list[str]]", classification_model) -> "tuple[list[int], list[int]]":
    """
    Trains and fits a classification model using a given corpus.

    Args:
        corpus (Tuple[List[str], List[int]]): A tuple containing a list of textual features and a list of corresponding labels.
        classification_model (Any): The classification model object to be trained and fitted.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing the true labels and predicted labels for the test dataset.
    """
    corpus_features = vectorize_set(corpus[1])
    corpus_labels = corpus[0]
    X_train, X_test, y_train, y_test = train_test_split(corpus_features, corpus_labels, test_size=0.2, random_state=42)
    model = classification_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred

def vectorize_set(corpus: "list[str]") -> "tuple[ndarray[spmatrix]]": # type: ignore
    """
    Vectorizes a given corpus using scikit-learn's CountVectorizer.
    Args:
        corpus (List[str]): A list of sentences or texts to be vectorized.
    Returns:
        Tuple[ndarray[spmatrix]]: A tuple containing the vectorized representation of the corpus.
    """
    # encode sentences as one-hot vectors using scikit-learn's one_hot function OHE
    vectorizer = sklfe.CountVectorizer()
    return vectorizer.fit_transform(corpus) # type: ignore