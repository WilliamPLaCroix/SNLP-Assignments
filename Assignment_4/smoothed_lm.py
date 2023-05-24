from collections import Counter

from nltk.util import ngrams


class BigramModel:
    def __init__(
        self, train_sents: "list[list[str]]", test_sents: "list[list[str]]", alpha=0
    ):
        """
        :param train_sents: list of sents from the train section of your corpus
        :param test_sents: list of sents from the test section of your corpus
        :param alpha :  Smoothing factor for laplace smoothing
        :function perplexity() : Calculates perplexity on the test set
        :function logprob(bigram) : Calculates the log probabillty of a bigram
        Tips:  Try out the Counter() module from collections to Count ngrams.
        """
        # TODO: Find unigram and bigram counts, extract ngrams from the test set for ppl computation
        self.alpha: int = alpha
        self.train_sents: list = train_sents
        self.test_sents: list = test_sents
        self.get_ngram_counts()

    def get_ngram_counts(self) -> None:
        self.train_unigram_counts = self.ngram_counter(self.train_sents, 1)
        self.train_bigram_counts = self.ngram_counter(self.train_sents, 2)
        self.test_unigram_counts = self.ngram_counter(self.test_sents, 1)
        self.test_bigram_counts = self.ngram_counter(self.test_sents, 2)
        return

    def ngram_counter(self, sents: "list[list[str]]", n: int) -> dict:
        ngram_counts = Counter()
        for sent in sents:
            ngram_counts.update(ngrams(sent, n))
        return dict(ngram_counts.most_common())

    def logprob(self, bigram: tuple):
        """returns the log proabability of a bigram. Adjust this function for Laplace Smoothing"""
        raise NotImplementedError

    def perplexity(self):
        """returns the average perplexity of the language model for the test corpus"""
        raise NotImplementedError
