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
from tensorflow.keras.utils import plot_model 
from matplotlib import pyplot as plt
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
MODELS_PATH = '../../models'

amino_acid_map = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'U': 18,'V': 19,'W': 20,'X': 21,'Y': 22,'Z': 23}


# ARGUMENTS PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='data directory')
parser.add_argument('--manifest_dir', type=str, default=MODELS_PATH, help='Models directory')




def create_model(arch='GRU', n_classes = 20, add_dropout=True, embeddings_dim=50, lr=1e-3, show_summary=True, manifest_dir=None):

    inp_recurrent = tf.keras.layers.Input([None])
    inp_conv = tf.keras.layers.Input([MAX_SEQ_LEN])
    # embed imput to an embedding dimension
    embedding_layer = tf.keras.layers.Embedding(input_dim=len(list(amino_acid_map.keys())), output_dim=embeddings_dim)
    if arch.lower() == 'gru':
        recurrent_layer = tf.keras.layers.GRU(units=256, dropout=.1, return_sequences=False)
    elif arch.lower() == 'lstm':
        recurrent_layer = tf.keras.layers.LSTM(units=256, dropout=.1, return_sequences=False)
    else:
        conv_layer = tf.keras.layers.Conv1D(128, 1, activation='relu')
        max_pool_layer = tf.keras.layers.MaxPooling1D(2)
        recurrent_layer = tf.keras.layers.GRU(units=256, dropout=.01, return_sequences=False)
        bidirectional_layer = tf.keras.layers.Bidirectional(recurrent_layer)
    # wrap reccurent layer with a bidirectional layer
    bidirectional_layer = tf.keras.layers.Bidirectional(recurrent_layer)

    # flatten layer
    flatten = tf.keras.layers.Flatten()

    # add fully connected layers
    fc1 = tf.keras.layers.Dense(units=256, activation="relu")
    # output layer 
    output_layer = tf.keras.layers.Dense(units=n_classes, activation="softmax")

    dropout_layer = tf.keras.layers.Dropout(0.05)

    # forward pass
    if arch.lower() =='cnn':
        embeds = embedding_layer(inp_conv)

        features = conv_layer(embeds)
        features = max_pool_layer(features)
        features = bidirectional_layer(features)
    else:
        embeds = embedding_layer(inp_recurrent)
        features = bidirectional_layer(embeds)

    features = flatten(features)
    features = fc1(features)
    features = dropout_layer(features)
  
    # get logits 
    logits = dropout_layer( output_layer(features))

    if arch.lower() =='cnn':
        model = tf.keras.Model(inputs=inp_conv, outputs=logits)
    else:
        model = tf.keras.Model(inputs=inp_recurrent, outputs=logits)

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', 
                    optimizer=opt,
                    metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
    
    if show_summary:
        model.summary()
        
    try:
        os.makedirs(manifest_dir, exist_ok=True)

        with open(os.path.join(manifest_dir, 'manifest.json'), 'w') as f:
            f.write(model.to_json())
            f.write('\n'+str(opt.get_config()))

    except:
        pass

    return model



def main(args):

    model = create_model(manifest_dir=args.manifest_dir, arch='CNN')


if __name__=='__main__':
    args = parser.parse_args()
    main(args)
