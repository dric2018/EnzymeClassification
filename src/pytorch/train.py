import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers, callbacks, seed_everything, Trainer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import transformers
import os
from transformers import AutoTokenizer, AutoModel

from datasets import EnzymeDataset
from models import EnzymeClassifier
import argparse
import sys 
import gc

from sklearn.model_selection import KFold, StratifiedKFold



###### Constants
data_dir = os.path.join('../data/TrainV1.csv')
models_dir = os.path.join('../models')
###### argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=data_dir,  type=str, help='data source directory')
parser.add_argument('--batch_size', default=16,  type=int, help='training batch_size')
parser.add_argument('--val_batch_size', default=8,  type=int, help='validation batch_size')
parser.add_argument('--epochs', default=60,  type=int, help='training epochs')
parser.add_argument('--n_folds', default=5,  type=int, help='number of cross validation splits')
parser.add_argument('--lr', type=float, default=5e-3,  help='Learning rate for model training')
parser.add_argument('--gpus', type=int, default=1,  help='Number of gpus to use for training')
parser.add_argument('--save_models_to', type=str, help='Directory to save trained models to')
parser.add_argument('--seed_value', default=60,  type=int, help='seed for reproducibility')



def make_folds(data:pd.DataFrame, args, n_folds = 10, target_col='label'):

    data['fold'] = 0

    fold = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=args.seed_value)
    for i, (tr, vr) in enumerate(fold.split(data, data[target_col])):
        data.loc[vr, 'fold'] = i

    return data, n_folds


def run_fold(fold, train_df, args, path=models_dir):
  
    torch.cuda.empty_cache()
    MAX_SEQ_LEN = 512
    INPUT_SIZE = 512

    fold_train = train_df[train_df.fold != fold].reset_index(drop=True)
    fold_val = train_df[train_df.fold == fold].reset_index(drop=True)

    train_ds = EnzymeDataset(df=fold_train, task='train', max_seq_len=MAX_SEQ_LEN)
    val_ds = EnzymeDataset(df=fold_val, task='train', max_seq_len=MAX_SEQ_LEN)

    trainloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True , num_workers=os.cpu_count())
    validloader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False , num_workers=os.cpu_count())

    del train_ds
    del val_ds
    del fold_train
    del fold_val

    
    model = EnzymeClassifier(input_size=INPUT_SIZE, 
                            hidden_size=128, 
                            max_seq_len=MAX_SEQ_LEN,
                            n_layers=10,
                            lr=args.lr,
                            n_classes=20)


    tb_logger = loggers.TensorBoardLogger(save_dir='../runs', name='ZINDI-Enzyme-classification', version=fold)

    ckpt_callback = pl.callbacks.ModelCheckpoint(filename=f'enzyme_classifier-fold-{fold}', 
                                                dirpath=path, 
                                                monitor='val_acc', 
                                                mode='max')
    
    trainer = Trainer(max_epochs=args.epochs, gpus=args.gpus, logger=tb_logger, callbacks=[ckpt_callback])

    trainer.fit(model, trainloader, validloader)


    gc.collect() # collect garbage

    return trainer.logged_metrics







if __name__=='__main__':
    
    args = parser.parse_args()
    _ = seed_everything(args.seed_value)
    train = pd.read_csv(args.data_path)
    train, n_folds = make_folds(n_folds=5, data=train, args=args, target_col='TARGET')

    # traiining loop
    best_fold = 0
    avg_log_loss = 0.0
    best_logloss = np.inf

    for fold in range(n_folds):

        print('')
        print('*'*18)
        print(f'Training on fold {fold}')
        print('*'*18)
        metrics = run_fold(fold=fold, train_df=train, args=args, path=args.save_models_to)
        #print(metrics)
        val_acc = metrics['val_acc']
        val_loss = metrics['val_loss']
        train_acc =metrics['train_acc_epoch']
        train_loss = metrics['train_loss_epoch']
        
        print('')
        print('*'*75)
        print(f'\t\t Results for Fold {fold}')
        print('-'*75)

        print(f'> Train Acc : \t{train_acc} \t| Valid Acc : {val_acc}')
        print(f'> Train logloss : {train_loss} \t| Valid logloss : {val_loss}')
        print('-'*75)
        print(f'\t\t Results for Fold {fold}')
        print('*'*75)
        if metrics['val_loss'] < best_logloss:
            best_logloss = metrics['val_loss']
            best_fold = fold
            avg_log_loss += metrics['val_loss']
        else:
            avg_log_loss += metrics['val_loss']

    print(f'[INFO] Training done ! Avg LogLoss : {avg_log_loss / n_folds}')

