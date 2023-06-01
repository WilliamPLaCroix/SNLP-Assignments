import string
from collections import Counter  # defaultdict

import nltk
import numpy as np
from nltk import ngrams

# nltk.download('treebank') # Uncomment this line if you have not downloaded the Treebank corpus


def flatten(lst):
    """Flattens a nested list into a single flat list.
    Args:
        lst (list): The nested list to flatten.
    Yields:
        Iterator: An iterator that yields the flattened items of the list.
    Example:
        >>> lst = [[1, 2], [3, [4, 5]], 6]
        >>> list(flatten(lst))
        [1, 2, 3, 4, 5, 6]
    """
    for item in lst:
        if isinstance(item, list):
            for nested_item in flatten(item):
                yield nested_item
        else:
            yield item


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


def split_corpus(corpus: "list[list[str]]", start: int, stop: int) -> "tuple[list[list[str]], list[list[str]]]":
    """
    Split the corpus into train and test sets.
    Args:
        corpus (list[list[str]]): The input corpus to be split.
    Returns:
        train: (list[list[str]]), test: (list[list[str]]) : A tuple containing the train and test set splits.
    """
    return corpus[:start] + corpus[stop:], corpus[start:stop]


class InterpolatedModel:
    """A language model that uses interpolated n-gram probabilities for prediction.
        This class represents a language model that calculates probabilities of n-grams and
        uses linear interpolation to estimate the probability of higher-order n-grams.
    Args:
        train (list[list[str]]): The training data, a list of sentences where each sentence is a list of words.
        test (list[list[str]]): The test data, a list of sentences where each sentence is a list of words.
        order (int): The order of the language model, specifying the number of words in an n-gram.
        alpha (float): The smoothing parameter used in calculating the conditional probabilities.
    Attributes:
        train (list[list[str]]): The training data.
        test (list[list[str]]): The test data.
        order (int): The order of the language model.
        alpha (float): The smoothing parameter.
        lambdar (float): The interpolation weight for each n-gram.
        train_ngram_counts (list[dict[tuple[str, ...], int]]): The n-gram counts for training data.
        test_ngram_counts (list[dict[tuple[str, ...], int]]): The n-gram counts for test data.
        train_context_counts (list[dict[tuple[str, ...], int]]): The context counts for training data.
        test_context_counts (list[dict[tuple[str, ...], int]]): The context counts for test data.
    Methods:
        ngram_counter(sents: list[list[str]], order: int, context: bool = False) -> dict[tuple[str, str], int]:
            Counts the n-grams in the data set.
        set_ngram_counts(order: int, context: bool = False) -> tuple[list[dict[tuple[str, ...], int]]]:
            Sets the n-gram counts for training and test data.
        conditional_probability(ngram: tuple[str, ...], order: int) -> float:
            Calculates the conditional probability of an n-gram.
        linear_interpolation(ngram: tuple[str, ...]) -> float:
            Calculates the linear interpolation probability of an n-gram.
        perplexity() -> float:
            Calculates the perplexity of the language model on a test set.
    """
    def __init__(self, train: "list[list[str]]", test: "list[list[str]]", order: int, alpha: float) -> None:
        """Initializes the language model with the provided training and test data.
        Args:
            train (list[list[str]]): The training data, a list of sentences where each sentence is a list of words.
            test (list[list[str]]): The test data, a list of sentences where each sentence is a list of words.
            order (int): The order of the language model, specifying the number of words in an n-gram.
            alpha (float): The smoothing parameter used in calculating the conditional probabilities.
        """
        self.train: "list[list[str]]" = train
        self.test: "list[list[str]]" = test
        self.order: int = order
        self.alpha: float = alpha
        self.lambdar: float = 1 / order
        self.train_ngram_counts, self.test_ngram_counts = self.set_ngram_counts(order)
        self.train_context_counts, self.test_context_counts = self.set_ngram_counts(order-1, context=True)

    
    def ngram_counter(self, sents: "list[list[str]]", order: int, context: bool = False) -> "dict[tuple[str, str], int]":
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
        #print("counting ngrams. order: ", order, " context: ", context)
        ngram_counts = Counter()
        for sent in sents:
            if context == True:
                sent = sent[:-1]
            ngram_counts.update(list(ngrams(sent, order)))
        return Counter(dict(ngram_counts.most_common()))


    def set_ngram_counts(self, order: int, context: bool = False) -> "tuple[list[dict[tuple[str, ...], int]], list[dict[tuple[str, ...], int]]]":
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
        #print("setting ngram counts. order:", order, "context:", context)
        return ([self.ngram_counter(self.train, i+1, context) for i in range(order)],
                 [self.ngram_counter(self.test, i+1, context) for i in range(order)])


    def conditional_probability(self, ngram: "tuple[str,...]", order: int) -> float:
        """Calculates the conditional probability of an n-gram.
            Conditional probability is the probability of an event (in this case, an n-gram) occurring
            given a certain context (n-1 gram). The formula used is:
            conditional_prob = (C(ngram) + alpha) / (C(context) + alpha * V)
            where V = size of vocabulary = len(unigram_counts)
        Args:
            ngram (tuple[str,...]): The n-gram for which the conditional probability is calculated.
            order (int): The order of the n-gram.
        Returns:
            float: The conditional probability of the given n-gram.
        """
        numerator = self.train_ngram_counts[order-1][ngram] + self.alpha
        try:
            denominator = self.train_context_counts[order-2][ngram[:-1]] + self.alpha * len(self.train_ngram_counts[0])
        except IndexError:
            denominator = sum(self.train_ngram_counts[0].values()) + self.alpha * len(self.train_ngram_counts[0])
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return 0

        
    def linear_interpolation(self, ngram: "tuple[str,...]") -> float:
        """Calculates the linear interpolation probability of an n-gram.
            Linear interpolation is a technique used to combine probabilities from lower-order n-grams
            to estimate the probability of a higher-order n-gram.
        Args:
            ngram (tuple[str,...]): The n-gram for which the linear interpolation probability is calculated.
        Returns:
            float: The linear interpolation probability of the given n-gram.
        """
        try:
            return np.log(self.lambdar * self.conditional_probability((ngram[0],), 1) + 
                      sum([self.lambdar * self.conditional_probability(ngram[:i+1], i+1) for i in range(1, self.order)]))
        except ValueError:
            return 0


    def perplexity(self) -> float:
        """Calculates the perplexity of the language model.
            Perplexity is a measure of how well a language model predicts a given test data set.
            It is calculated using the formula PP = exp(-sum(f(w,h)*log(P(w|h))).
            Where f(w,h) = relative frequency of bigram in test data
            And P(w|h) = conditional probability of bigram in training data
        Args:
            None
        Returns:
            float: The perplexity score of the language model.
        """
        N = sum(self.test_ngram_counts[self.order-1].values())

        dalist = list()
        for ngram in self.test_ngram_counts[self.order-1]:
            dalist.append((self.test_ngram_counts[self.order-1][ngram] / N) * self.linear_interpolation(ngram))

        return np.exp(-sum(dalist))
        # return np.exp(-sum(
        #     [(self.test_ngram_counts[self.order-1][ngram] / N) * self.linear_interpolation(ngram)
        #         for ngram in self.test_ngram_counts[self.order-1]]))
    