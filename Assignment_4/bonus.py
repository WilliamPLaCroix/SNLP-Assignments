def char_tokenize(corpus):
    char_corpus = []
    for sent in corpus:
        sent_tokenized = [c for c in " ".join(sent)]
        char_corpus.append(sent_tokenized)
    return char_corpus
