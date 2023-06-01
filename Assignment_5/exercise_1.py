import string
from collections import Counter  # defaultdict

import nltk
#import numpy as np
from nltk import ngrams

# nltk.download('treebank') # Uncomment this line if you have not downloaded the Treebank corpus

def load_and_preprocess_data() -> "list[list[str]]":
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
    # for sent in sents: # Remove sentences with only one word. Not sure if we'll need this.
    #     if len(sent) == 1:
    #         corpus.remove(sent)
    return [" ".join(sent).lower().translate(str.maketrans('', '', string.punctuation)).split() 
            for sent in list(nltk.corpus.treebank.sents())]

def make_vocab(corpus: "list[list[str]]", top_n: int) -> "list[str]":
    """
    Creates a vocabulary of the most frequent words in the given corpus.
    Args:
        corpus (list[list[str]]): A list of lists, where each inner list represents a sentence
            in the corpus.
        top_n (int): The number of words to include in the vocabulary.
    Returns:
        List[str]: The vocabulary, which is a list of the most frequent words in the corpus.
    """
    vocab = dict()
    for sent in corpus:
        for word in sent:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_sorted = sorted(list(vocab.items()), key=lambda x: x[1], reverse=True)[:top_n]

    return [v[0] for v in vocab_sorted]

def restrict_vocab(corpus: "list[list[str]]", V=5000) -> "list[list[str]]":
    """
    Restricts the vocabulary of the given corpus to a specified size.
    Args:
        corpus (list[list[str]]): A list of lists, where each inner list represents a sentence
            in the corpus.
    Returns:
        list[list[str]]: The restricted corpus, where words not present in the vocabulary are
            replaced with "<unk>".
    """
    vocab: "list[str]" = make_vocab(corpus, V)
    corpus_restrict: "list[list[str]]" = []
    for sent in corpus:
        sent_restrict: "list[str]" = []
        for word in sent:
            if word not in vocab:
                sent_restrict.append("<unk>")
            else:
                sent_restrict.append(word)
        corpus_restrict.append(sent_restrict)
    return corpus_restrict

# use a 70:30 split for train/test. Take test from the last 30%. Do not randomize at this stage.
def split_corpus(corpus: "list[list[str]]"):
    """
    Split the corpus into train and test sets.
    Args:
        corpus (list[list[str]]): The input corpus to be split.
    Returns:
        train: (list[list[str]]), test: (list[list[str]]) : A tuple containing the train and test set splits.
    """
    return corpus[:int(len(corpus)*0.7)], corpus[int(len(corpus)*0.7):]


class InterpolatedModel():
    """#TODO"""
#Complete this class
    def _init_(self, train, test, order=2, alpha=0.5) -> None:
        """#TODO"""
        self.train: "list[list[str]]" = train
        self.test: "list[list[str]]" = test
        self.order: int = order
        self.alpha: float = alpha
        self.train_ngram_counts, self.test_ngram_counts = self.set_ngram_counts(order)
    
    def ngram_counter(self, sents: "list[list[str]]", order: int) -> "dict[tuple[str, str], int]":
        """
        Counts the n-grams in the given sentences.
        Args:
            sents (List[List[str]]): A list of lists, where each inner list represents a sentence
                to count the n-grams from.
            order (int): The order of the n-grams to count (1 for unigrams, 2 for bigrams, etc.)
        Returns:
            Dict[str, int]: A dictionary containing the counts of the n-grams, where the keys
                are the n-grams represented as strings, and the values are the corresponding counts.
        Example:
            >>> obj = MyClass()
            >>> sents = [['this', 'is', 'an', 'example', 'sentence'], ['another', 'example', 'sentence']]
            >>> obj.ngram_counter(sents, 2)
            {'this is': 1, 'is an': 1, 'an example': 1, 'example sentence': 2, 'another example': 1}
        """
        ngram_counts = Counter()
        for sent in sents:
            if order == 1:
                sent = sent[:-1]
            ngram_counts.update(list(ngrams(sent, order)))
        return Counter(dict(ngram_counts.most_common()))

    def set_ngram_counts(self, order: int) -> "tuple[list[dict[tuple[str, ...], int]], list[dict[tuple[str, ...], int]]]":
        """
        Sets the n-gram counts for training and test data for each n up to the given 'order'.
        Args:
            order (int): The maximum order of n-grams to count.
        Returns:
            Tuple[List[Dict[Tuple[str, ...], int]], List[Dict[Tuple[str, ...], int]]]: A tuple containing the n-gram counts
                for each order up to 'order' for the training and test data, respectively. Each element of the tuple is a
                list of dictionaries, where each dictionary represents the n-gram counts for a specific order. The keys of
                the dictionaries are tuples representing the n-grams, and the values are the corresponding counts.
        Example:
            >>> model = InterpolatedModel()
            >>> model.set_ngram_counts(3)
            (train[{unigram_counts}, {bigram_counts}, {trigram_counts}], test[{unigram_counts}, {bigram_counts}, {trigram_counts}])
        """
        return ([self.ngram_counter(self.train, i) for i in range(order)],
                 [self.ngram_counter(self.test, i) for i in range(order)])

    def logprob(self, bigram: "tuple[str,...]") -> float:
        """#TODO"""
        # PLaplace(w_n|w_n-1) = (C(w_n-1,w_n) + alpha)
        #                       (C(w_n-1) + alpha * V)
        # V = size of vocabulary = len(unigram counts)
        # numerator = self.train_bigram_counts[bigram] + self.alpha
        # denominator = self.train_unigram_counts[(bigram[0],)] + self.alpha * len(self.train_unigram_counts)
        # try:
        #     return np.log(numerator / denominator)
        # except ValueError:
        #     return -100000000000000000000000000000000
        raise NotImplementedError
        
    def perplexity(self, order: int) -> float:
        """#TODO"""
        # PP = exp(-sum(f(w,h)*log(P(w|h)))
        # f(w,h) = relative frequency of bigram in test data
        # P(w|h) = conditional probability of bigram in training data
        # N = sum(self.test_bigram_counts.values())
        # return np.exp(-sum(
        #     [(self.test_bigram_counts[bigram] / N) * self.logprob(bigram)
        #         for bigram in self.test_bigram_counts]))
        raise NotImplementedError
    