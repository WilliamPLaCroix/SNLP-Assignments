from collections import defaultdict, Counter
import nltk
import string
import re

nltk.download('treebank')


def load_and_preprocess_data():
    ''' Function that loads the WSJ dataset, removes punctuation and lowercases the tokens'''
    corpus = list(nltk.corpus.treebank.sents())
    preprocessed_corpus = []
    # TODO: Preprocess by removing punctuations and lowercasing.
    # Remove all sentences with <2 tokens, we can't use bigram models on these.
    '''For preprocessing, remove tokenized punctuations and punctuations within words. String punctuations can be 
    found in `string.punctuations`.'''
    for sent in corpus:
        sent_preprocessed = []
        for word in sent:
            word = word.lower()
            word = re.sub(r'[^\w\s]', '', word)
            if word:  # skip empty strings
                sent_preprocessed.append(word)
        if len(sent_preprocessed) >= 2:
            preprocessed_corpus.append(sent_preprocessed)

    return preprocessed_corpus


def train_test_split(corpus):
    '''Splits the corpus using a 70:30 ratio. Do not randomize anything here. use the original order
  Input: List[List[str]]
  Output: List[List[str]],List[List[str]]'''
    corpus_size = len(corpus)
    split_index = int(corpus_size * 0.7)
    return corpus[:split_index], corpus[split_index:]


def make_vocab(corpus, top_n):
    '''Make the top_n frequent vocabulary from a corpus
  Input: corpus - List[List[str]]
         top_n  - int
  Output: Vocabulary - List[str]'''
    vocab = dict()
    for sent in corpus:
        for word in sent:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_sorted = sorted(list(vocab.items()), key=lambda x: x[1], reverse=True)[:top_n]

    return [v[0] for v in vocab_sorted]


def restrict_vocab(corpus, vocab):
    '''Make the corpus fit inside the vocabulary using <unk>
  Input: corpus - List[List[str]]
         vocab  - List[str]
  Output: Vocabulary_restricted_corpus - List[List[str]]'''
    corpus_restrict = []
    for sent in corpus:
        sent_restrict = []
        for word in sent:
            if word not in vocab:
                sent_restrict.append("<unk>")
            else:
                sent_restrict.append(word)
        corpus_restrict.append(sent_restrict)

    return corpus_restrict
