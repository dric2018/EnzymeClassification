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

from utils import get_data_loader


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
amino_acid_map = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'U': 18,'V': 19,'W': 20,'X': 21,'Y': 22,'Z': 23}


# ARGUMENTS PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='data directory')


def create_model(arch='GRU', n_classes = 20, add_dropout=True, embeddings_dim=256, lr=1e-3):

    inp = tf.keras.layers.Input([None])
    # embed imput to an embedding dimension
    embedding_layer = tf.keras.layers.Embedding(input_dim=len(list(amino_acid_map.keys())), output_dim=embeddings_dim)
    if arch.lower() == 'gru':
        recurrent_layer = tf.keras.layers.GRU(units=300, dropout=0.2, return_sequences=False)
    else:
        recurrent_layer = tf.keras.layers.LSTM(units=300, dropout=.2, return_sequences=False)
    
    # wrap reccurent layer with a bidirectional layer
    bidirectional_layer = tf.keras.layers.Bidirectional(recurrent_layer)

    # add fully connected layers
    fc1 = tf.keras.layers.Dense(units=1024, activation="relu")
    fc2 = tf.keras.layers.Dense(units=512, activation="relu")
    fc3 = tf.keras.layers.Dense(units=64, activation="relu")

    # output layer 
    output_layer = tf.keras.layers.Dense(units=n_classes, activation="softmax")

    dropout_layer = tf.keras.layers.Dropout(0.3)

    # forward pass

    embeds = embedding_layer(inp)
    features = bidirectional_layer(embeds)
    features = fc1(features)
    if dropout:
        features = dropout_layer(features)
    features = fc2(features)
    features = fc3(features)
    # get logits 
    logits = output_layer(features)

    model = tf.keras.Model(inputs=inp, outputs=logits)

    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss='sparse_categorical_crossentropy', 
                    optimizer=opt,
                    metrics=['accuracy'])
    
    print(model.summary())

    return model



def main(args):

    model = get_model()


if __name__=='__main__':
    args = parser.parse_args()
    main(args)
