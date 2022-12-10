#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 
import tensorflow as tf

from typing import List

from collections import defaultdict
from collections import OrderedDict
import re
import string
import math

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    lexemes = wn.lemmas(lemma, pos=pos) # Retrieve all lexemes for the input lemma and input PoS
    lemmas = []
    if lexemes:
        for l in lexemes:
            synset=l.synset()
            lem=synset.lemmas()
            for elem in lem:
                lemmas.append(elem)

        candidate_subs = {l.name().replace("_", " ") for l in lemmas}
        candidate_subs.remove(lemma)
        return candidate_subs
    else:
        return {}

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
# Total = 298, attempted = 298
# precision = 0.094, recall = 0.094
# Total with mode 206 attempted 206
# precision = 0.136, recall = 0.136
    lemma = context.lemma
    pos = context.pos
    lexemes = wn.lemmas(lemma, pos=pos) # Retrieve all lexemes for the input lemma and input PoS
    lemmas = []
    syn_dict = defaultdict(int)
    if lexemes:
        for l in lexemes:
            synset=l.synset()
            lem=synset.lemmas()
            for elem in lem:
                lemmas.append(elem)
    
    for k in lemmas:
        key = k.name().replace("_", " ")
        if key != lemma:
            syn_dict[key] = k.count()
    
    from collections import OrderedDict
    syn_dict = OrderedDict(sorted(
                    syn_dict.items(),
                    key = lambda x: x[1],
                    reverse = True))
    
    max_syn = max(syn_dict, key = lambda x: syn_dict[x])
    return max_syn

def wn_simple_lesk_predictor(context : Context) -> str:
# Total = 298, attempted = 298
# precision = 0.076, recall = 0.076
# Total with mode 206 attempted 206
# precision = 0.112, recall = 0.112    
    lemma = context.lemma
    pos = context.pos
    synsets = wn.synsets(lemma, pos)
    lexemes = wn.lemmas(lemma, pos=pos)
    left_context = context.left_context
    right_context = context.right_context
    full_context = left_context + right_context
    stop_words = stopwords.words('english')
    filt_context = [word.lower() for word in full_context 
             if word not in stop_words and word.isalpha()]
    overlap_dict = defaultdict(int)
    max_overlap = 0
    
    for syn in synsets:
        defn = syn.definition()
        for ex in syn.examples():
            defn += " " + ex
        hypernyms = syn.hypernyms()
        for hyp in hypernyms:
            defn += " " + hyp.definition()
            for hyp_ex in hyp.examples():
                defn += " " + hyp_ex
        tokens = [tok for tok in tokenize(defn) if tok not in stop_words]
        
        overlap_words = [tok for tok in tokens for word in filt_context if tok==word]
        num_overlap = len(overlap_words)
        overlap_dict[syn] = num_overlap
        if num_overlap > max_overlap:
            max_overlap = num_overlap
    
    # Determine if there are any ties: 
    max_count = 0
    for k in overlap_dict.values():
        if k == max_overlap: max_count +=1
            
    if max_overlap==0 or max_count>1:
        return wn_frequency_predictor(context)
    else:
        max_overlap_synset = max(overlap_dict, key = lambda x: overlap_dict[x])
        max_syn_name = max_overlap_synset.lemmas()[0].name().replace('_',' ')
        return max_syn_name       
   

class Word2VecSubst(object):
# Total = 298, attempted = 298
# precision = 0.115, recall = 0.115
# Total with mode 206 attempted 206
# precision = 0.170, recall = 0.170        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        cand_syns = get_candidates(context.lemma, context.pos)
        # Obtain a vector representation for each candidate synonym
        w2v_vectors = self.model.wv  
        best_score = 0.0
        most_sim = ''
#         for syn in cand_syns:
#              # Not all synonyms are in Word2Vec
#             if syn in w2v_vectors:
#                 sim_score = self.model.similarity(context.lemma, syn)
#                 if sim_score > best_score:
#                     best_score = self.model.similarity(context.lemma, syn)
#                     most_sim = syn
        sim_score_dict = {}
        for syn in cand_syns:
             # Not all synonyms are in Word2Vec
            if syn in w2v_vectors:
                sim_score_dict[syn] = self.model.similarity(context.lemma, syn)
        most_sim = max(sim_score_dict, key = lambda x: sim_score_dict[x])  
        
        return most_sim 


class BertPredictor(object):
# Total = 298, attempted = 298
# precision = 0.115, recall = 0.115
# Total with mode 206 attempted 206
# precision = 0.170, recall = 0.170
    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Obtain the set of WordNet-derived candidate synonyms:
        cand_syns = get_candidates(context.lemma, context.pos)
        
        # Convert context information into masked input representation for the DistilBERT model:
        mask = ['[MASK]']
        full_context = context.left_context + mask + context.right_context
        context_string = ' '.join(full_context)
        input_toks = self.tokenizer.encode(context_string)
        mask_ind = self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        input_mat = np.array(input_toks).reshape((1,-1))
        
        # Run the DistilBERT model on the input representation constructed above:
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        best_tok_ids = np.argsort(predictions[0][mask_ind])[::-1] 
        best_words = self.tokenizer.convert_ids_to_tokens(best_tok_ids, skip_special_tokens=False)
        best_words = [word.replace("_", " ") for word in best_words]
        # From the list of highest-scoring words, isolate those that are contained within the set of
        # WordNet-derived candidate synonyms:
        best_synonyms = [syn for syn in best_words if syn in cand_syns]
            
        # In case best_words does not contain any words in set of synonyms from get_candidates(), 
        # use the target word itself as the replacement.
        highest_scoring_word = context.lemma  
        
        # Extract highest scoring candidate synonym:
        if best_synonyms:
            highest_scoring_word = best_synonyms[0] 
#         for tok in best_words:
#             if tok in cand_syns:
#                 highest_scoring_word = tok
#                 break
        return highest_scoring_word 

# Part 6 Predictor:
class P6_Predictor(object):   
    "Using a combination of both the Word2Vec model and the DistilBERT model such that the cosine similarity scores are combined with the log-probabilities from the BERT model. Specifically, the similarity score for the target word and the list of words from BERT are computed. Then, for each word, the corresponding similarity score is then scaled up and added to the log-probability from BERT. The word with the highest overall score is selected." 
# Total = 298, attempted = 298
# precision = 0.128, recall = 0.128
# Total with mode 206 attempted 206
# precision = 0.189, recall = 0.189
    def __init__(self, filename): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        
    def predict(self, context : Context) -> str:
        # Obtain the set of WordNet-derived candidate synonyms:
        cand_syns = get_candidates(context.lemma, context.pos)
        
        # Convert context information into masked input representation for the DistilBERT model:
        mask = ['[MASK]']
        full_context = context.left_context + mask + context.right_context
        context_string = ' '.join(full_context)
        input_toks = self.tokenizer.encode(context_string)
        mask_ind = self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        input_mat = np.array(input_toks).reshape((1,-1))
        
        # Run the DistilBERT model on the input representation constructed above:
        outputs = self.bert_model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        ## sort probs in increasing order and return token IDs (or indices)
        best_tok_ids = np.argsort(predictions[0][mask_ind])[::-1]
        ## Sort predicted values using sorted ids above
        sorted_probs = predictions[0][mask_ind][best_tok_ids]
        best_words = self.tokenizer.convert_ids_to_tokens(best_tok_ids, skip_special_tokens=False)
        best_words = [word.replace("_", " ") for word in best_words]
        
#         probs = np.zeros(len(sorted_probs))
#         for i in range(len(sorted_probs)):
#             probs[i] = math.exp(sorted_probs[i])
        
        w2v_vectors = self.w2v_model.wv
        sim_scores = np.zeros(len(best_words))
        for word in best_words:
            if word in w2v_vectors:
                idx = best_words.index(word)
                sim_scores[idx] = self.w2v_model.similarity(context.lemma, word)
                sim_scores[idx] *= 10
            else:
                idx = best_words.index(word)
                sim_scores[idx] = 0.0001
        
        score_dict = {}
        for i, word in enumerate(best_words):
            if word in cand_syns:
                score_dict[word] = sim_scores[i]+sorted_probs[i]
            else:
                score_dict[word] = 0

        # Extract highest scoring candidate synonym:
        top_score_word = max(score_dict, key = lambda x: score_dict[x])

        return top_score_word
    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = '../GoogleNews-vectors-negative300.bin'
# #     predictor = Word2VecSubst(W2VMODEL_FILENAME) # Part 4 predictor
# #     predictor = BertPredictor()                  # Part 5 predictor
    predictor = P6_Predictor(W2VMODEL_FILENAME)  # Part 6 predictor  

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
#         # Part 2 WordNet Frequency Model:
#         prediction = wn_frequency_predictor(context)
#         # Part 3 WordNet Lesk Algorithm Model:
#         prediction = wn_simple_lesk_predictor(context)
#         # Part 4 Word2Vec Predictor:
#         prediction = predictor.predict_nearest(context) 
#         # Part 5 BERT Predictor:
#         prediction = predictor.predict(context) 
        # Part 6 Predictor:
        prediction = predictor.predict(context) 
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
