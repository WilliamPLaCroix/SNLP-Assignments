# TODO: Add your necessary imports here
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import FreqDist, bigrams
from nltk.corpus import brown
from math import exp


def load_corpus():
    """Load `Brown` corpus from NLTK"""
    return [word.lower() for word in brown.words()]


def get_relative_freqs(text: list):
    """Get bigram frequencies for the provided text.

    Args:
    text -- A `list` containing the tokenized text to be
            used to calculate the frequencies of bigrams
    """
    bigram_counts = FreqDist(bigrams(text))
    bigram_freqs = []
    for bigram in bigram_counts:
        bigram_freqs.append((bigram, bigram_counts[bigram] / (len(text) - 1)))
    return bigram_freqs


def get_bigram_probs(text: list):
    """ Get bigram frequencies for the provided text.

    Args:
    text -- A `list` containing the tokenized text to be
            used to calculate the frequencies of bigrams
    """
    bigram_probs = dict()
    unigram_probs = dict()
    for i in range(len(text) - 1):
        w1 = text[i]
        w2 = text[i + 1]
        bigram = (w1, w2)
        if bigram in bigram_probs:
            bigram_probs[bigram] += 1
        else:
            bigram_probs[bigram] = 1
        if w1 in unigram_probs:
            unigram_probs[w1] += 1
        else:
            unigram_probs[w1] = 1

    for bigram in bigram_probs:
        bigram_probs[bigram] /= unigram_probs[bigram[0]]
    return bigram_probs


def get_top_n_probabilities(text: list, context_words: Union[str, list], n: int):
    """Get top `n` following words to `context_words` and their probabilities

    Args:
    text -- A list of tokens to be used as the corpus text
    context_words -- A `str` containing the context word(s) to be considered
    n    -- An `int` that indicates how many tokens to evaluate
    """
    top_n = {}
    bigram_frequency_distribution = FreqDist(bigrams(text))
    for word in context_words:
        norm = 0
        filtered_bigram_frequency_distribution = []
        for bigram in bigram_frequency_distribution:
            if bigram[0] == word:
                norm += bigram_frequency_distribution[bigram]
        for bigram in bigram_frequency_distribution:
            if bigram[0] == word:
                filtered_bigram_frequency_distribution.append(
                    (bigram, bigram_frequency_distribution[bigram] / norm)
                )
        top_n[word] = filtered_bigram_frequency_distribution[0:n]
    return top_n


def get_entropy(top_n_dict: dict, relative_freqs):
    """Get entropy of distribution of top `n` bigrams"""
    entropy_dict = dict()
    for word in top_n_dict:
        relative_freqs_filtered = []
        for bigram, probs in relative_freqs:
            if bigram in [b[0] for b in top_n_dict[word]]:
                relative_freqs_filtered.append((bigram, probs))
        word_cond_probs = np.array([p[1] for p in top_n_dict[word]])
        relative_freqs_array = np.array([f[1] for f in relative_freqs_filtered])
        entropy_dict[word] = - np.sum(np.multiply(relative_freqs_array, np.log2(word_cond_probs)))
    return entropy_dict


def plot_top_n(top_n_dict: dict):
    """Plot top `n`"""
    for context in top_n_dict:
        dataframe = pd.DataFrame(top_n_dict[context])

        X = np.array(range(len(dataframe.index)))  # use ranks for x values
        X = X + 1  # indexing needs to start from 1 for log scale
        Y = np.array(list(dataframe[1]))

        plt.bar(X, Y)
        plt.ylabel("absolute frequency")
        plt.xlabel("rank")
        plt.title(f"'{context}'")
        plt.show()


def get_perplexity(phrases, relative_freqs, cond_freqs):
    relative_freqs_filtered = []
    cond_freqs_filtered = []
    for phrase in phrases:
        bigram = tuple(phrase.split())
        for r_freq in relative_freqs:
            if r_freq[0] == bigram:
                relative_freqs_filtered.append(r_freq[1])
        if bigram in cond_freqs:
            cond_freqs_filtered.append(cond_freqs[bigram])

    return exp(-np.sum(np.multiply(relative_freqs_filtered, np.log2(cond_freqs_filtered))))


def get_mean_rank(bigram_list: list, text: list):
    context, word = bigram_list[0].split()
    print(context)
    filtered_bigram_frequency_distribution = []
    bigram_frequency_distribution = get_relative_freqs(text)
    for bigram, count in bigram_frequency_distribution:
        if bigram[0] == context:
            key = f"{bigram[0]} {bigram[1]}"
            filtered_bigram_frequency_distribution.append((key, count))
    dataframe = pd.DataFrame(filtered_bigram_frequency_distribution)
    ranks = []

    for bigram in bigram_list:
        try:
            ranks.append(dataframe[dataframe[0] == bigram].index.values[0] + 1)
        except IndexError:
            continue
        except KeyError:
            continue
    return print("Mean rank:", int(np.mean(ranks)))
