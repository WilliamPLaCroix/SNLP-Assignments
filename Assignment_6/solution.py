import string

import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as sklfe
from nltk import ngrams
from nltk.corpus import stopwords
from scipy.sparse import spmatrix  # pylint: disable=unused-import
from sklearn.model_selection import train_test_split

# nltk.download('stopwords')
# sentiments_to_indexes: dict = {"Positive": "2", "Neutral": "1", "Negative": "0"}
# indexes_to_sentiments: dict = {"2": "Positive", "1": "Neutral", "0": "Negative"}

def calculate_confusion_matrix(y_true: "np.ndarray", y_pred: "np.ndarray") -> "pd.DataFrame":
    """
    Create a confusion matrix from two lists of labels.
    Args:
        y_true (list[int]): The true sentiment labels.
        y_pred (list[int]): The predicted sentiment labels.
    Returns:
        np.ndarray: 3x3 confusion matrix.
    """
    indexes_to_sentiments: dict = {"2": "Positive", "1": "Neutral", "0": "Negative"}
    y_pred = np.array([indexes_to_sentiments[str(elem)] for elem in y_pred])
    y_true = np.array([indexes_to_sentiments[str(elem)] for elem in y_true])
    return (pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
            .reindex(columns=["Negative", "Neutral", "Positive"],
                       index=["Negative", "Neutral", "Positive"], fill_value=0))


def load_and_preprocess_data(path: str="./all-data.csv", ngramize: int=1, polarize: bool=False,
                            remove_stops: bool=False) -> "list[str]":
    """
    Load and preprocess data from a CSV file.
    Args:
        path (str, optional): The path to the CSV file. Defaults to "./all-data.csv".
        ngramize (int, optional, default: 1): The value specifying the degree of n-gramization.
        polarize (bool, optional, default: False): Flag indicating whether to polarize the corpus. 
        remove_stops (bool, optional, default: False): Flag indicating whether to remove stopwords from the corpus.                   
    Returns:
        list[str]: A list of preprocessed sentences, where each sentence is represented as a list of words.
    """
    dataframe = pd.read_csv(path, encoding="ISO-8859-1", header=None)
    data = dataframe[1].tolist()
    corpus: list = []
    for sent in data:
        sent = sent.split()
        sent = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent]
        sent = [word.lower() for word in sent]
        corpus.append(" ".join(" ".join(sent).split())) # list[str] where each str is a sentence
    if remove_stops:
        corpus = remove_stopwords(corpus)
    if polarize:
        corpus = polarize_corpus(corpus, load_loughlan_dict())
    if ngramize != 1:
        corpus = ngramize_corpus(corpus, ngramize)
    dataframe[0] = replace_labels_with_numbers(dataframe[0])
    dataframe[1] = corpus
    return dataframe


def load_loughlan_dict(path: str="./Loughran-McDonald_MasterDictionary_1993-2021.csv") -> "dict[str, dict[str, str]]":
    """
    Load the Loughran-McDonald sentiment dictionary from the given CSV file.
    Args:
        path (str): The path to the CSV file containing the sentiment dictionary.
    Returns:
        dict[str, dict[str, str]]: A nested dictionary representing the sentiment dictionary, 
                                    where each word is mapped to its corresponding sentiment polarity.
    """
    sentiment_dataframe = pd.read_csv(path, encoding = "ISO-8859-1")
    sentiment_dataframe["Word"] = sentiment_dataframe["Word"].str.lower()
    sentiment_dictionary = sentiment_dataframe.set_index("Word").to_dict("index")
    return sentiment_dictionary


def ngramize_corpus(corpus: "list[str]", order: int = 2) -> "list[str]":
    """
    Generate n-grams from the given corpus.
    Args:
        corpus (list[str]): The corpus containing preprocessed sentences.
        order (int, optional): The degree of n-gramization. Defaults to 2.
    Returns:
        list[str]: A list of n-grams, where each n-gram is represented as a tuple of words.
    """
    ngramized_corpus = []
    for sentence in corpus:
        ngramized_sentence = list(ngrams(sentence.split(), n=order))
        joined_ngrams: "list[str]" = []
        for ngram in ngramized_sentence:
            joined_ngrams.append("".join([word for word in ngram]))
        new_sentence = " ".join(joined_ngrams)
        ngramized_corpus.append(new_sentence)
    return ngramized_corpus


def polarize_corpus(corpus: "list[str]", sentiment_dictionary) -> "list[str]":
    """
    Polarize the given corpus using a sentiment dictionary.
    Args:
        corpus (list[str]): The corpus containing preprocessed sentences.
        sentiment_dictionary (dict[str, dict[str, str]]): A sentiment dictionary mapping words to their polarities.
    Returns:
        list[str]: A list of polarized sentences, where each word is represented as an integer polarity value.
    """
    polarized_corpus = []
    for sentence in corpus:
        polarized_sentence = []
        for word in sentence.split():
            try:
                word_polarity = sentiment_dictionary[word]
                if word_polarity["Positive"] != "0":
                    polarized_sentence.append("Positive")
                elif word_polarity["Negative"] != "0":
                    polarized_sentence.append("Negative")
                else:
                    polarized_sentence.append("Neutral")
            except KeyError:
                polarized_sentence.append("Neutral")
        polarized_corpus.append(" ".join(polarized_sentence))
    return polarized_corpus


def remove_stopwords(corpus: "list[str]") -> "list[str]":
    """
    Remove stopwords from the given corpus.
    Args:
        corpus (list[str]): The corpus containing preprocessed sentences.
    Returns:
        list[str]: A list of sentences with stopwords removed.
    """
    stopword_set = set(stopwords.words('english'))
    cleaned_corpus = []
    rm_stopword_count = 0
    total_word_count = 0
    for sentence in corpus:
        new_sentence = []
        for word in sentence.split():
            total_word_count += 1
            if word not in stopword_set:
                new_sentence.append(word)
            else:
                rm_stopword_count += 1
        cleaned_corpus.append(" ".join(new_sentence))
    print("total word count:", total_word_count)
    print("removed stopword count:", rm_stopword_count)
    print("ratio", rm_stopword_count/total_word_count)
    return cleaned_corpus


def replace_labels_with_numbers(dataframe: "list[str]") -> "list[str]":
    """
    Replace labels in the corpus with numerical representations.
    Args:
        corpus (list[str]): The corpus containing labeled data.
    Returns:
        list[str]: The corpus with labels replaced by their corresponding numerical representations.
    """
    sentiments: dict = {"Positive": 2, "positive": 2, "Neutral": 1, "neutral": 1, "Negative": 0, "negative": 0}
    dataframe = [sentiments[str(elem)] for elem in dataframe]
    return dataframe


def test(confusion_matrix, classifier: str, preprocessing: str) -> "dict[str, str|float]":
    """
    Calculate and print evaluation metrics based on the provided confusion matrix.
    Args:
        confusion_matrix (pd.DataFrame): A pandas DataFrame representing the confusion matrix.
        classifier (str): The name of the classifier used.
        preprocessing (str): The name of the preprocessing technique used.
    Returns:
        None
    """
    TP = confusion_matrix['Positive']['Positive']
    TNeutral = confusion_matrix['Neutral']['Neutral']
    TNegative = confusion_matrix['Negative']['Negative']
    FP = confusion_matrix['Negative']['Positive'] + confusion_matrix['Neutral']['Positive']
    FN = confusion_matrix['Neutral']['Negative'] + confusion_matrix['Positive']['Negative']
    Total = (confusion_matrix['Negative']['Negative'] + confusion_matrix['Negative']['Neutral'] + confusion_matrix['Negative']['Positive'] +
                confusion_matrix['Neutral']['Negative'] + confusion_matrix['Neutral']['Neutral'] + confusion_matrix['Neutral']['Positive'] +
                confusion_matrix['Positive']['Negative'] + confusion_matrix['Positive']['Neutral'] + confusion_matrix['Positive']['Positive'])

    # Accuracy = (TP + TNeut + TNeg) / (Total)
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

    accuracy = (TP + TNeutral + TNegative) / Total
    print(f"\n{preprocessing} {classifier} accuracy: {accuracy:.2f}")
    precision = TP / (TP + FP)
    print(f"{preprocessing} {classifier} precision: {precision:.2f}")
    recall = TP / (TP + FN)
    print(f"{preprocessing} {classifier} recall: {recall:.2f}")
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
    print(f"{preprocessing} {classifier} F1: {F1:.2f}")
    return {"Model": f"{preprocessing} {classifier}", "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": F1}

def train_and_fit_model(corpus: "tuple[list[str|int], list[str]]", classification_model) -> "tuple[list[int], list[int]]":
    """
    Trains and fits a classification model using a given corpus.
    Args:
        corpus (tuple[list[str], list[int]]): A tuple containing a list of textual features and a list of corresponding labels.
        classification_model (Any): The classification model object to be trained and fitted.
    Returns:
        tuple[list[int], list[int]]: A tuple containing the true labels and predicted labels for the test dataset.
    """
    corpus_features = vectorize_set(corpus[1])
    corpus_labels = corpus[0]
    X_train, X_test, y_train, y_test = train_test_split(corpus_features, corpus_labels, test_size=0.2, random_state=42)
    model = classification_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred


def vectorize_set(corpus: "list[str]") -> "np.ndarray|spmatrix":
    """
    Vectorizes a given corpus using scikit-learn's CountVectorizer.
    Args:
        corpus (list[str]): A list of sentences or texts to be vectorized.
    Returns:
        "np.ndarray|spmatrix": array/matrix containing the vectorized representation of the corpus.
    """
    vectorizer = sklfe.CountVectorizer()
    return vectorizer.fit_transform(corpus)
