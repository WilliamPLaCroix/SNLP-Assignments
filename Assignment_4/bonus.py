from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def char_tokenize(corpus):
    char_corpus = []
    for sent in corpus:
        sent_tokenized = [c for c in " ".join(sent)]
        char_corpus.append(sent_tokenized)
    return char_corpus


def byte_pair_tokenize(corpus):
    byte_pair_corpus = []
    for sent in corpus:
        sent_tokenized = [str(token) for token in tokenizer(" ".join(sent))["input_ids"]]
        byte_pair_corpus.append(sent_tokenized)
    return byte_pair_corpus
