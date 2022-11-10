import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
Joseph High (UNI: jph2185)
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    word_list = sequence.copy()
    start = ['START']
    stop = ['STOP']
    # Adding the appropriate number of 'START' elements to the beginning of the list and 
    # one 'STOP' the end of the list.
    if n==1:
        word_list = start + word_list
        word_list.extend(['STOP'])
    else:
        word_list = start*(n-1) + word_list
        word_list.extend(stop)
    
    # Creates a list of n-gram tuples from the input sequence
    ngram_list = [x for x in zip(*[word_list[i:] for i in range(n)])]
    
    return ngram_list

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # Counts total number of sentences:
        generator = corpus_reader(corpusfile, self.lexicon)
        sent_count = 0
        for sentence in generator:
            sent_count +=1
        self.num_sentences = sent_count
        
        # Counts total number of words:
        generator = corpus_reader(corpusfile, self.lexicon)
        total_word_count = 0
        for sentence in generator:
            for word in sentence:
                total_word_count +=1
        # Adding number of sentences to initial word count to account for 'STOP' in each sentence:
        self.total_num_words = total_word_count + self.num_sentences
        
        
    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        for lst in corpus:
            
            self.unigrams = get_ngrams(lst,1)
            self.bigrams = get_ngrams(lst,2)
            self.trigrams = get_ngrams(lst,3)
        
            for unigram in self.unigrams:
                self.unigramcounts[unigram] += 1

            for bigram in self.bigrams:
                self.bigramcounts[bigram] += 1

            for trigram in self.trigrams:
                self.trigramcounts[trigram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        count_uvw = self.trigramcounts[trigram]
        count_uv = self.bigramcounts[trigram[0:2]]
        
        # initialize variable for iteration
        raw_trigram_prob = float()
        
        # As advised by Professor Bauer, P(START, START, word) should be treated as follows:
        if trigram[0:2] == ('START', 'START'):
            raw_trigram_prob = count_uvw/self.num_sentences
        elif count_uv==0:
            raw_trigram_prob = 1/len(self.lexicon) # 1/|V| when denominator = 0
        else:
            raw_trigram_prob = count_uvw/count_uv
            
        return raw_trigram_prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """  
        count_uw = self.bigramcounts[bigram]
        count_u = self.unigramcounts[bigram[0:1]]
        
        # initialize variable for iteration
        raw_bigram_prob = float()
    
        if count_u==0:
            raw_bigram_prob = 1/len(self.lexicon) # 1/|V| when denominator = 0
        else:
            raw_bigram_prob = count_uw/count_u
            
        return raw_bigram_prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
#         raw_unigram_prob = float()
#         if self.unigramcounts[unigram] == 0:
#             raw_unigram_prob = 1/self.total_num_words
#         else:
#             raw_unigram_prob = self.unigramcounts[unigram]/self.total_num_words
        raw_unigram_prob = self.unigramcounts[unigram]/self.total_num_words

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return raw_unigram_prob

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        trigram_prob = self.raw_trigram_probability(trigram)
        bigram_prob = self.raw_bigram_probability(trigram[1:3])
        unigram_prob = self.raw_unigram_probability(trigram[2:3])
        
        smooth_prob = lambda1*trigram_prob + lambda2*bigram_prob + lambda3*unigram_prob
        
        return smooth_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        log_prob = float()
        
        for tgram in trigrams:
            log_prob += math.log2(self.smoothed_trigram_probability(tgram))
            
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        M = 0
        num_sentences = 0
        probsum = float()
        for sentence in corpus:
            num_sentences +=1
#             sentence.extend(['STOP'])
            probsum += self.sentence_logprob(sentence)
            for word in sentence:
                M += 1
        M = M + num_sentences # Adding # of sentences to account for 'STOP' in each sentence. 
            
        l = (1/M)*probsum
        pp = 2**(-l)
        
        return pp


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
        
        # Assumes that training_file1 == 'train_high.txt' and testdir1 == 'test_high'
        for f in os.listdir(testdir1): 
            pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp_model1 < pp_model2:
                correct+=1
            total+= 1
            
        # Assumes that training_file1 == 'train_low.txt' and testdir2 == 'test_low'
        for f in os.listdir(testdir2): 
            pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if pp_model1 > pp_model2:
                correct+=1
            total+= 1 
            
        accuracy = correct/total
        
        return accuracy

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 
    
    # Training Perplexity:
    train_corpus = corpus_reader(sys.argv[1], model.lexicon)
    pp_train = model.perplexity(train_corpus)
    print('Training Perplexity:', pp_train)
    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print('Testing Perplexity:', pp)


    # Essay scoring experiment: 
#     acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")
    path = 'hw1_data/ets_toefl_data/'
    acc = essay_scoring_experiment(path+'train_high.txt', path+'train_low.txt', 
                         path+'test_high/', path+'test_low/')
    print('Accuracy on Essay Data:', acc)

