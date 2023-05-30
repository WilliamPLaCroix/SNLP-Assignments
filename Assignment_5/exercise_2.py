import string


def preprocess_text(text: str) -> "list[str]":
    """
    Preprocesses the given text by removing punctuation, converting to lowercase,
    and splitting it into tokens.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list[str]: A list of tokens obtained after removing punctuation, converting
            to lowercase, and splitting the input text.

    Example:
        >>> text = "Hello, World!"
        >>> preprocess_text(text)
        ['hello', 'world']
    """
    return text.translate(str.maketrans('', '', string.punctuation)).lower().split()

def KneserNeyCounter():
    """#TODO"""
    #Implement required counting functions to compute bigram probability
    raise NotImplementedError