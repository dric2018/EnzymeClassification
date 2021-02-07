# IMPORTS
import pytorch_lightning as pl

import torch as th
from torch.utils.data import DataLoader, Dataset

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from .config import Config
from .tokenizer import EnzymeTokenizer

# CONSTANTS AND VARIABLES

class EnzymeDataset(Dataset):
    def __init__(self, df:pd.DataFrame, 
                 config:Config=Config,task='train', *args, **kwargs):
        super(EnzymeDataset,self).__init__()

        self.task = task
        self.df = df
        self.n_classes = config.num_classes
        self.config = config
        self.letters_to_int = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'U': 18,'V': 19,'W': 20,'X': 21,'Y': 22,'Z': 23}
        self.max_seq_len = config.max_seq_len

        self.tokenizer = EnzymeTokenizer(vocab = Config.default_vocab)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get row data
        data = dict(self.df.iloc[index])
        sample_data = {
            'seq' : data['SEQUENCE']
        }

        code = self.tokenizer.encode(
                sequence = data['SEQUENCE'], 
                max_len=self.max_seq_len, 
                padding=True
            )
            
        sample_data = {
                'input_ids' : code['ids'],
            }
        try:
            target = data['TARGET']
            sample_data.update({
                'trg' : th.tensor(target, dtype=th.long)
            })
        except :
            pass

        return sample_data


    
class DataModule(pl.LightningDataModule):

    def __init__(self, config: Config, 
                 train_df:pd.DataFrame,
                 validation_split=.25,
                test_df:pd.DataFrame = None,
                 train_frac = 1):

        super(DataModule, self).__init__()
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.train_df = train_df 
        self.test_df = test_df 
        self.validation_split = validation_split
        self.train_frac = train_frac
        self.num_workers = config.num_workers
        
    def setup(self, stage=None):
        
        if self.train_frac > 0 and self.train_frac < 1 :
            self.train_df = self.train_df.sample(frac=self.train_frac).reset_index(drop=True)
            train, val = train_test_split(self.train_df, 
                                          test_size=self.validation_split, 
                                          random_state=Config.seed_val)
        else:
            train, val = train_test_split(self.train_df, 
                                          test_size=self.validation_split, 
                                          random_state=Config.seed_val)
            
        self.train_ds = EnzymeDataset(
            df=train, 
            task='train'
        )

        self.val_ds = EnzymeDataset(
            df=val, 
            task='train'
)

        print(f'[INFO] Training on {len(self.train_ds)}')
        print(f'[INFO] Validating on {len(self.val_ds)}')

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers
                         )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers
                         )