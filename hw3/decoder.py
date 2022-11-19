from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State
# Decoding took awhile, so turning off "eager execution" for tensorflow (as recommended)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            # TODO: Write the body of this loop for part 4 
            
            # Instance of feature extractor to obtain rep of current state:
            model_feat = self.extractor.get_input_representation(words, pos, state).reshape((1, 6))
            # Call model and retrieve vector of possible actions
            pred_actions = np.argsort(self.model.predict(model_feat)[0])[::-1]
            output = self.output_labels
            
            for i in pred_actions:
                transition, label = output[i]
                if transition == 'right_arc':
                    if not state.stack: # right-arc not permitted if stack is empty
                        continue
                    elif state.stack:
                        state.right_arc(label)
                        break  
                elif transition=='left_arc':  
                    if not state.stack:  # left-arc not permitted if stack is empty
                        continue
                    elif state.stack[-1]==0: # Root can't be target of left_arc
                        continue
                    else:
                        state.left_arc(label)
                        break
                elif transition=='shift': 
                    # Shifts are illegal if buffer only has one word, unless the stack is empty
                    if not state.stack or len(state.buffer)>1: 
                        state.shift()
                        break
                        
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
