"""
COMS W4705 - Natural Language Processing - Summer 2022 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        import numpy as np
        import math
        
        n = len(self.lhs_to_rules)
        bool_list = np.zeros(n)
        total_probs = np.zeros(n)
        result = False
        upper_result, prob_result, rhs_len_result = False, False, False
        
        # Checking that probabilities sum to 1 for each nonterminal and that 
        # nonterminal symbols are uppercase:
        for i, key in enumerate(self.lhs_to_rules):
            prob_sum=0
            for j in range(len(self.lhs_to_rules[key])):
                prob_sum += self.lhs_to_rules[key][j][2]
            total_probs[i] = prob_sum
            if key.isupper() == True:
                bool_list[i] += 1
                
        # Verifying that the RHS has 2 nonterminals or 1 terminal, aligning with CNF:  
        rhs_bool = np.zeros(len(self.rhs_to_rules))
        for i, key in enumerate(self.rhs_to_rules):
            for j in range(len(self.rhs_to_rules[key])):
                if 1<=len(self.rhs_to_rules[key][j][1])<=2:
                    rhs_bool[i] = 1
                    
        if math.isclose(sum(total_probs), len(self.lhs_to_rules)):
            prob_result=True
        if sum(bool_list)==len(self.lhs_to_rules):
            upper_result=True
        if sum(rhs_bool)==len(self.rhs_to_rules):
            rhs_len_result=True
        
        if prob_result and upper_result and rhs_len_result:
            result=True
        
        return result 


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        
