import string
from collections import Counter, defaultdict

import nltk

# nltk.download('treebank') # Uncomment this line if you have not downloaded the Treebank corpus

def load_and_preprocess_data() -> "list[list[str]]":
    """
    Loads and preprocesses the data from the NLTK Treebank corpus.

    Returns:
        List[List[str]]: A list of lists, where each inner list represents a sentence
            from the Treebank corpus after removing punctuation, converting to lowercase,
            and removing empty tokens.

    Example:
        >>> load_and_preprocess_data()
        [['this', 'is', 'an', 'example', 'sentence'], ['another', 'example', 'sentence']]
    """
    corpus = list(nltk.corpus.treebank.sents())
    # Remove punctuation and lowercase, removing empty tokens after cleaning
    return [" ".join(sent).lower().translate(str.maketrans('', '', string.punctuation)).split() for sent in corpus]

class Interpolated_Model():
    """#TODO"""
#Complete this class
    def _init_(self, args):
        """#TODO"""
        raise NotImplementedError

    def perplexity(self, args):
        """#TODO"""
        raise NotImplementedError
    