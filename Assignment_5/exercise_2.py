import string
from nltk import ngrams
from collections import Counter


def preprocess_text(text: str) -> "list[str]":
    """
    Preprocesses the given text by removing punctuation, converting to lowercase,
    and splitting it into tokens.
    Args:
        text (str): The input text to be preprocessed.
    Returns:
        list[str]: A list of tokens obtained after removing punctuation, converting
            to lowercase, and splitting the input text.
    """
    return text.translate(str.maketrans('', '', string.punctuation)).lower().split()


def get_ngram_counts(corpus, ngram_size=3):
    """#TODO"""
    # Implement required counting functions to compute bigram probability
    corpus_ngrams = list(ngrams(corpus, ngram_size))
    ngram_counts = Counter(corpus_ngrams)

    return ngram_counts


def get_unigram_counts(corpus):
    return Counter(corpus)


def get_w2o_counts(bigrams):
    count_dict = dict()
    for w2, _ in bigrams:
        try:
            count_dict[w2] += 1
        except KeyError:
            count_dict[w2] = 1
    return Counter(count_dict)


def get_ow2w3_counts(trigrams):
    count_dict = dict()
    for _, w2, w3 in trigrams:
        try:
            count_dict[(w2, w3)] += 1
        except KeyError:
            count_dict[(w2, w3)] = 1
    return Counter(count_dict)


def get_ow2o_counts(trigrams):
    count_dict = dict()
    for _, w2, _ in trigrams:
        try:
            count_dict[w2] += 1
        except KeyError:
            count_dict[w2] = 1
    return Counter(count_dict)


def get_ow3_counts(bigrams):
    count_dict = dict()
    for _, w3 in bigrams:
        try:
            count_dict[w3] += 1
        except KeyError:
            count_dict[w3] = 1
    return Counter(count_dict)


def lamda_w2(w2, d, unigram_counts, w2o_counts):
    return d / unigram_counts[w2] * w2o_counts[w2]


def P_KN_w3(w3, unigram_counts, ow3_counts):
    if w3 in unigram_counts:
        return ow3_counts[w3] / len(ow3_counts)
    else:
        return 1 / len(unigram_counts)


def P_KN_w3_w2(d, w2, w3, ow3_counts, ow2w3_counts, ow2o_counts, unigram_counts, w2o_counts):
    return max(ow2w3_counts[(w2, w3)] - d, 0) / ow2o_counts[w2] + \
        lamda_w2(w2, d, unigram_counts, w2o_counts) * P_KN_w3(w3, unigram_counts, ow3_counts)


class TrigramModel:
    def __init__(self, corpus, d):
        self.d = d
        self.trigram_counts = get_ngram_counts(corpus, ngram_size=3)
        self.bigram_counts = get_ngram_counts(corpus, ngram_size=2)
        self.unigram_counts = get_unigram_counts(corpus)
        self.w2o_counts = get_w2o_counts(self.bigram_counts)
        self.ow2w3_counts = get_ow2w3_counts(self.trigram_counts)
        self.ow3_counts = get_ow2o_counts(self.trigram_counts)
        self.ow2o_counts = get_ow2o_counts(self.trigram_counts)

    def get_lamda_w2(self, w2):
        return lamda_w2(w2, self.d, self.unigram_counts, self.w2o_counts)

    def get_P_KN_w3(self, w3):
        return P_KN_w3(w3, self.unigram_counts, self.ow3_counts)

    def get_P_KN_w3_w2(self, w2, w3):
        return P_KN_w3_w2(self.d, w2, w3, self.ow3_counts, self.ow2w3_counts, self.ow2o_counts,
                          self.unigram_counts, self.w2o_counts)
