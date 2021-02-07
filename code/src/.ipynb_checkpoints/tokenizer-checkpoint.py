import torch as th 
import torch.nn as nn
import numpy as np

from .config import Config

# function to return key for any value
def get_key(val):
    """
    from : https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
    """
    for key, value in Config.default_vocab.items():
         if val == value:
            return key
        
    return "key doesn't exist"



class EnzymeTokenizer:
    def __init__(self, vocab=Config.default_vocab):
        self.vocab = vocab    
    
    
    def encode(self, sequence, padding=True, max_len=512):
        if len(sequence) > max_len:
            tokens = ' '.join(sequence[:max_len]).split()
        else:
            tokens = ' '.join(sequence).split()
        padding_len = np.abs(max_len - len(tokens))
        
        ids = [self.vocab[token] for token in tokens] + [24]*padding_len
        # mask = th.zeros(max_len)
        # mask[:len(tokens)] = 1

        return {
            'ids': th.tensor(ids, dtype=th.long),
            # 'mask': mask.float()
        }
    
    
    def decode(self, ids):
        sequence = ''.join([get_key(ids[index]) for index in range(len(ids))])
        return sequence




