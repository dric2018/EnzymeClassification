import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd 
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import argparse
import sys 


###### Constants
data_dir = os.path.join('../data/TrainV1.csv')

###### argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=data_dir,  type=str, help='data source directory')

class EnzymeDataset(Dataset):
    def __init__(self, df:pd.DataFrame, task='train',max_seq_len=512, n_classes=20, *args, **kwargs):
        super(EnzymeDataset,self).__init__()

        self.task = task
        self.df = df
        self.n_classes = n_classes
        self.letters_to_int = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'U': 18,'V': 19,'W': 20,'X': 21,'Y': 22,'Z': 23}
        self.max_seq_len = max_seq_len


    def __len__(self):
        return len(self.df)


    def convert_letters_to_int(self, sequence:str, mapping:dict, max_seq_len:int):
 
        if len(sequence) > max_seq_len:
            sequence = sequence[:max_seq_len]
        else:
            pad_len = max_seq_len - len(sequence)
            sequence += "".join("A" * pad_len)

        int_seq = torch.tensor([[mapping[l] for l in sequence]], dtype=torch.long)

        return int_seq


    def __getitem__(self, index):

        # get row data
        data = dict(self.df.iloc[index])
        
        sample_data = {
            'sequence' :  self.convert_letters_to_int(sequence = data['SEQUENCE'], mapping=self.letters_to_int, max_seq_len=self.max_seq_len),
        }
        try:
            target = data['TARGET']
            sample_data.update({
                'target' : torch.tensor(target, dtype=torch.long)
            })
        except :
            pass

        return sample_data



if __name__ == '__main__':
    args = parser.parse_args()
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 600
    train = pd.read_csv(args.data_path)
    ds = EnzymeDataset(df=train, task='train', max_seq_len=MAX_SEQ_LEN)

    dl = DataLoader(dataset=ds, 
                    batch_size=BATCH_SIZE,
                    num_workers=os.cpu_count())

    for data in dl:
        xs, ys = data['sequence'], data['target']
        print(xs.shape)
        print(ys.shape)
        break