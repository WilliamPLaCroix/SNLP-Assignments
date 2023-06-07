import string

import nltk
import pandas as pd
import sklearn.feature_extraction.text as sklfe
import sklearn.metrics
from numpy import ndarray
from sklearn.model_selection import train_test_split


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

def load_and_preprocess_data(path: str="./all-data.csv", remove_punct: bool=True, lowercase: bool=True) -> "list[list[str]]":
    """
    Load and preprocess data from a CSV file.
    Args:
        path (str, optional): The path to the CSV file. Defaults to "./all-data.csv".
        remove_punct (bool, optional): Flag indicating whether to remove punctuation from the data. Defaults to True.
        lowercase (bool, optional): Flag indicating whether to convert the data to lowercase. Defaults to True.
    Returns:
        List[List[str]]: A list of preprocessed sentences, where each sentence is represented as a list of words.
    """
    dataframe = pd.read_csv(path, encoding="ISO-8859-1", header=None)
    data = dataframe[1].tolist()
    corpus = list()
    for sent in data:
        sent = sent.split()
        # if remove_punct:
        sent = [word.translate(str.maketrans('', '', string.punctuation)) for word in sent]
        # if lowercase:
        sent = [word.lower() for word in sent]
        corpus.append(" ".join(" ".join(sent).split()))
    dataframe[1] = corpus
    return dataframe

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

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"\n{preprocessing} {classifier} accuracy:", accuracy)
    precision = TP / (TP + FP)
    print(f"{preprocessing} {classifier} precision:", precision)
    recall = TP / (TP + FN)
    print(f"{preprocessing} {classifier} recall:", recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"{preprocessing} {classifier} F1:", f1)
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
    X_train, X_test, y_train, y_test = train_test_split(corpus_features, corpus_labels, test_size=0.2)#, random_state=42)
    model = classification_model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred

# def train_test_split(corpus: "list[list[str]]", train_ratio: float = 0.8) -> "tuple[list[list[str]], list[list[str]]]":
#     """
#     Split the corpus into train and test sets using a 80:20 ratio..
#     Args:
#         corpus (list[list[str]]): The input corpus to be split.
#         train_ratio (float): The ratio of the train set to the test set.
#     Returns:
#         train: (list[list[str]]), test: (list[list[str]]) : A tuple containing the train and test set splits.
#     """
#     return corpus[:int(len(corpus) * train_ratio)], corpus[int(len(corpus) * train_ratio):]