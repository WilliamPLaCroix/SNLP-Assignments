from collections import defaultdict,Counter
import nltk
import string
nltk.download('treebank')

def load_and_preprocess_data():
  corpus=list(nltk.corpus.treebank.sents())
  cleaned_corpus=[]
  for sent in corpus:
    #Remove punctuations
    words = [word.translate(str.maketrans(dict.fromkeys(string.punctuation))).lower() for word in sent]
    #Remove empty tokens after cleaning
    words=" ".join(words).split()
    if len(words)>=2:
      cleaned_corpus.append(words)
  return cleaned_corpus

  class Interpolated_Model():
    #Complete this class
    def _init_(args):
      raise NotImplementedError

    def perplexity(args):
      raise NotImplementedError