import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import pandas as pd 
import transformers

from transformers import AutoTokenizer, AutoModel



class EnzymeClassifier(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(EnzymeClassifier, self).__init__()
        self.save_hyperparameters()

    def forwrd(self, x):
        pass


    def training_step(self, batch, batch_idx):
        pass



    def validation_step(self, batch, batch_idx):
        pass


    def test_step(self, batch, batch_idx):
        pass


    def get_loss(self, logits, targets):
        pass


    def get_acc(self, logits, targets):
        pass