from collections import Counter
from math import log

import numpy as np
from nltk.util import ngrams


class BigramModel:
    """
    :function perplexity() : Calculates perplexity on the test set
    :function logprob(bigram) : Calculates the log probabillty of a bigram
    Tips:  Try out the Counter() module from collections to Count ngrams.
    """

    def __init__(
        self, train_sents: "list[list[str]]", test_sents: "list[list[str]]", alpha=0
    ):
        """
        :param train_sents: list of sents from the train section of your corpus
        :param test_sents: list of sents from the test section of your corpus
        :param alpha :  Smoothing factor for laplace smoothing
        """
        self.alpha: float = alpha
        self.train_sents: list = train_sents
        self.test_sents: list = test_sents
        self.get_ngram_counts()

    def get_ngram_counts(self) -> None:
        """calls ngram_counter for unigrams and bigrams for each data set"""
        self.train_unigram_counts = self.ngram_counter(self.train_sents, 1)
        self.train_bigram_counts = self.ngram_counter(self.train_sents, 2)
        self.test_unigram_counts = self.ngram_counter(self.test_sents, 1)
        self.test_bigram_counts = self.ngram_counter(self.test_sents, 2)
        return

    def ngram_counter(self, sents: "list[list[str]]", n: int) -> dict:
        """returns a dict containing counts of n-grams from a given set"""
        ngram_counts = Counter()
        for sent in sents:
            ngram_counts.update(list(ngrams(sent, n)))
        return dict(ngram_counts.most_common())

    def logprob(self, bigram: tuple) -> float:
        """returns the log proabability of a bigram. Adjust this function for Laplace Smoothing"""
        # PLaplace(w_n|w_n-1) = (C(w_n-1,w_n) + alpha)
        #                       (C(w_n-1) + alpha * V)
        # V = size of vocabulary = len(unigram counts)
        try:
            numerator = self.train_bigram_counts[bigram] + self.alpha
        except KeyError:
            numerator = self.alpha

        denominator = self.train_unigram_counts[(bigram[0],)] + self.alpha * len(self.train_unigram_counts)
        try:
            return log(numerator / denominator)
        except ValueError:
            return 1

    def perplexity(self):
        """returns the average perplexity of the language model for the test corpus"""
        # PP = exp(-sum(f(w,h)*log(P(w|h)))
        # f(w,h) = relative frequency of bigram in test data
        # P(w|h) = conditional probability of bigram in training data
        N = sum(self.test_bigram_counts.values())
        return np.exp(-sum(
            [(self.test_bigram_counts[bigram] / N) * self.logprob(bigram)
                for bigram in self.test_bigram_counts]))
