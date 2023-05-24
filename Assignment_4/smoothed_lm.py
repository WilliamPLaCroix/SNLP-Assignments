#Add necessary imports here
class BigramModel:
    
    def __init__(self, train_sents: List[List[str]], test_sents: List[List[str]], alpha=0):
        """ 
        :param train_sents: list of sents from the train section of your corpus
        :param test_sents: list of sents from the test section of your corpus
        :param alpha :  Smoothing factor for laplace smoothing
        :function perplexity() : Calculates perplexity on the test set
        :function logprob(bigram) : Calculates the log probabillty of a bigram 
        Tips:  Try out the Counter() module from collections to Count ngrams. 
        """
        self.alpha = alpha
        #TODO: Find unigram and bigram counts, extract ngrams from the test set for ppl computation

    def logprob(self,bigram):
      '''returns the log proabability of a bigram. Adjust this function for Laplace Smoothing'''
      raise NotImplementedError

    def perplexity(self):
        """ returns the average perplexity of the language model for the test corpus """
        raise NotImplementedError