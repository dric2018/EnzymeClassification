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

# RPRODUCIBILITY
seed_val = 2020
os.environ['PYTHONHASHSEED']=str(seed_val)
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.x
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


# UTILS
    
def seq_to_text_file(save_file_to:str, sequences, max_len=MAX_SEQ_LEN):
    with open(save_file_to, 'w') as f:
        for seq in senquences:
            if len(seq) > max_len:
                seq = "".join(list(seq)[0:max_len])
            f.write("%s\n" % seq)


def encode_sequence(text_tensor, label):
    encoded_text = [ amino_acid_map[e] for e in list(text_tensor.numpy().decode())]
    return encoded_text, label


def set_encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label=tf.one_hot(label,number_of_class)
    label.set_shape([number_of_class])
    
    return encoded_text, label


def get_data_loader(file,batch_size, labels, task='train'):
    
    label_data=tf.data.Dataset.from_tensor_slices(labels)
    data_set=tf.data.TextLineDataset(file)
    data_set=tf.data.Dataset.zip((data_set,label_data))

    if task=='train':
        data_set=data_set.repeat()
        data_set = data_set.shuffle(len(labels))

    data_set=data_set.map(encode_map_fn,tf.data.experimental.AUTOTUNE)
    data_set=data_set.padded_batch(batch_size)
    data_set = data_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return data_set






def run_fold(fold_num, model, dataset, save_ckpt_to, epochs, data_dir, log_dir='../runs/logs', train_bs=1024, val_bs=512):

    train = dataset[dataset["FOLD"] !=fold].reset_index(drop=True)
    validation = dataset[dataset["FOLD"] ==fold].reset_index(drop=True)

    # get labels 
    train_labels = train.TARGET
    validation_labels = validation.TARGET

    # create txt files 
    seq_to_text_file(save_file_to=os.path.join(data_dir, 'train_data_{fod_num}.txt'), sequences=train.SEQUENCE)
    seq_to_text_file(save_file_to=os.path.join(data_dir, 'validation_data_{fod_num}.txt'), sequences=validation.SEQUENCE)
    
    # callbacks 
    model_name = "enzyme_classifier"
    ckpt_cb_1 = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_ckpt_to, model_name),
                                                 save_weights_only=True,
                                                 monitor = 'val_acc',
                                                 save_best_only=True,
                                                 mode="max", 
                                                 verbose=1)

    ckpt_cb_2 = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_ckpt_to, model_name),
                                                 save_weights_only=True,
                                                 monitor = 'val_loss',
                                                 save_best_only=True,
                                                 mode="min",
                                                  verbose=1)

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', 
                                                    factor=0.1, patience=10, 
                                                    verbose=1,
                                                    mode='auto')

    es = tf.keras.callbacks.EarlyStopping(monitor="val_acc", 
                                            mode="max", 
                                            patience=10,
                                            restore_best_weights=True)

    tb_cb = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=0, write_graph=True,
                write_images=False, write_steps_per_second=False, update_freq='epoch',
                profile_batch=2, embeddings_freq=0, embeddings_metadata=None
            )


    CALL8BACKS = [ckpt_cb_1, ckpt_cb_2, lr_reducer, es, tb_cb]
                            
    # getting data loaders
    train_data_loader = get_data_loader(file=os.path.join(data_dir, 'train_data_{fod_num}.txt'), batch_size=train_bs, labels=train_labels, task='train')
    validation_data_loader = get_data_loader(file=os.path.join(data_dir, 'validation_data_{fod_num}.txt'), batch_size=val_bs, labels=validation_labels, task='validation')

    # train model

    history = model.fit(train_data_loader,
                validation_data = validation_data_loader,
                epochs=epochs,
                batch_size=train_bs,
                validationçbatch_size=val_bs,
                steps_per_epoch=len(train) // train_bs,
                validation_steps = len(validation) // val_bs,
                callbacks=CALLBACKS)

                
    return history


def test():
    pass


def make_predictions():
    pass


def make_folds(dataset:pd.DataFrame, n_folds=5, target_col='LABEL'):
    dataset['fold'] = -1
    kf = StratifiedKFold(n_splits=n_folds, random_state=seed_val, shuffle=True)

    for i, (tr, vr) in enumerate(kf.split(dataset, dataset[target_col])):
        dataset.loc[vr, 'fold'] = i

    return n_folds, dataset





# MAIN

def main(args):
    print(f"[INFO] using tensorflow version {tf.__version__}")

    train = pd.read_csv(os.path.join(args.data_path,'Train.csv'))
    train['LENGTH'] = train['SEQUENCE'].swifter.progress_bar(enable=True, desc='Computing sequence length').apply(lambda seq: len(seq))
    train['TARGET'] = train['LABEL'].swifter.progress_bar(enable=True, desc='Creating target column').apply(lambda c: int(c.split('class')[-1]))

    print(train.head())








if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
