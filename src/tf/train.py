import os
import pandas as pd
import swifter
import numpy as np
import sys
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
import argparse
import warnings
import tensorflow as tf

from utils import get_data_loader, make_folds, run_fold
from models import create_model


# RPRODUCIBILITY
seed_val = 2020
os.environ['PYTHONHASHSEED']=str(seed_val)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.x
tf.random.set_seed(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)

#### Env config 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings(action='ignore')

# CONSTANTS

MAX_SEQ_LEN = 600
DATA_PATH = '../../data'
MODELS_PATH = '../../models'
LOGS_PATH = '../../runs/logs'
amino_acid_map = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'U': 18,'V': 19,'W': 20,'X': 21,'Y': 22,'Z': 23}


# ARGUMENTS PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='data directory')
parser.add_argument('--train_bs', default=128, type=int, help='training batch size')
parser.add_argument('--validation_bs', default=64, type=int, help='validation batch size')
parser.add_argument('--n_folds', default=5, type=int, help='Number of splits for k-fold cross-validation')
parser.add_argument('--lr', default=2e-3, type=float, help='training learning rate')
parser.add_argument('--ckpt_dir', default=MODELS_PATH, type=str, help='Checkpoint directory')
parser.add_argument('--num_epochs', default=5, type=int, help='Number of training epochs')
parser.add_argument('--log_dir', default=LOGS_PATH, type=str, help='log directory')
parser.add_argument('--arch', default='CNN', type=str, help='Model architecture')



# training loop

def train_fn(args):
    best_fold = 0
    avg_acc = 0.0
    best_acc = 0.0

    dataset = pd.read_csv(os.path.join(args.data_path, 'TrainV1.csv'))

    if args.n_folds >=2:
        for fold in range(args.n_folds):
            model = create_model(show_summary=False, lr=args.lr, manifest_dir=args.ckpt_dir, arch=args.arch)

            print('')
            print('*'*18)
            print(f'Training on fold {fold}')
            print('*'*18)
            metrics = run_fold(fold_num=fold, 
                                model=model,
                                dataset = dataset,
                                save_ckpt_to=args.ckpt_dir,
                                epochs=args.num_epochs,
                                data_dir=args.data_path,
                                log_dir=args.log_dir,
                                train_bs=args.train_bs,
                                val_bs=args.validation_bs)

            val_acc = np.array(metrics['val_accuracy']).mean()
            val_loss = np.array(metrics['val_loss']).mean()
            train_acc = np.array(metrics['accuracy']).mean()
            train_loss =  np.array(metrics['loss']).mean()
            
            print('')
            print('*'*75)
            print(f'\t\t Results for Fold {fold}')
            print('-'*75)

            print(f'> Train Acc : \t{train_acc} \t| Valid Acc : {val_acc}')
            print(f'> Train logloss : {train_loss} \t| Valid logloss : {val_loss}')
            print('-'*75)
            print(f'\t\t Results for Fold {fold}')
            print('*'*75)

            if val_acc > best_acc:
                best_acc = val_acc
                best_fold = fold
                avg_acc += val_acc
            

            del model

        print()    
        print(f'[INFO] Training done ! \n(CV-LB) Avg Acc : {avg_acc / args.n_folds}')
        print(f'[INFO] Best fold : {best_fold}')
        print(f'[INFO] Best Accuracy : {best_acc}')

    else:
        model = create_model(show_summary=True, lr=args.lr, manifest_dir=args.ckpt_dir, arch=args.arch)

        print('')
        print('*'*18)
        print(f'  Training ')
        print('*'*18)
        metrics = run_fold(fold_num=args.n_folds, 
                            model=model,
                            dataset = dataset,
                            save_ckpt_to=args.ckpt_dir,
                            epochs=args.num_epochs,
                            data_dir=args.data_path,
                            log_dir=args.log_dir,
                            train_bs=args.train_bs,
                            val_bs=args.validation_bs)

        val_acc = np.array(metrics['val_accuracy']).mean()
        val_loss = np.array(metrics['val_loss']).mean()
        train_acc = np.array(metrics['accuracy']).mean()
        train_loss =  np.array(metrics['loss']).mean()
        
        print('')
        print('*'*75)
        print(f'\t\t Results at the end')
        print('-'*75)

        print(f'> Train Acc : \t{train_acc} \t| Valid Acc : {val_acc}')
        print(f'> Train logloss : {train_loss} \t| Valid logloss : {val_loss}')
        print('-'*75)
        print(f'\t\t Results ')
        print('*'*75)

        if val_acc > best_acc:
            best_acc = val_acc
            avg_acc += val_acc
        

        del model




if __name__=="__main__":
    args = parser.parse_args()
    train_fn(args)
