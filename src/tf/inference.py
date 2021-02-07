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

from utils import get_data_loader, make_folds, run_fold, seq_to_text_file, encode_sequence
from models import create_model
from collections import Counter

# RPRODUCIBILITY
seed_val = 2020
os.environ['PYTHONHASHSEED']=str(seed_val)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # TF 2.x
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
#Level | Level for Humans | Level Description                  
#-------|------------------|------------------------------------ 
# 0     | DEBUG            | [Default] Print all messages       
# 1     | INFO             | Filter out INFO messages           
# 2     | WARNING          | Filter out INFO & WARNING messages 
# 3     | ERROR            | Filter out all messages  
# ---------------------------------------------------------------


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
TEST_DATA_DIR = '../../data/test'
SUBMISSIONS_DIR = '../../submissions'
# ARGUMENTS PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--train_bs', default=256, type=int, help='training batch size')
parser.add_argument('--validation_bs', default=128, type=int, help='validation batch size')
parser.add_argument('--data_path', type=str, default=DATA_PATH, help='data directory')
parser.add_argument('--n_folds', default=5, type=int, help='Number of splits for k-fold cross-validation')
parser.add_argument('--lr', default=2e-3, type=float, help='training learning rate')
parser.add_argument('--ckpt_dir', default=MODELS_PATH, type=str, help='Checkpoint directory')
parser.add_argument('--log_dir', default=LOGS_PATH, type=str, help='log directory')
parser.add_argument('--files_dest_directory', default=TEST_DATA_DIR, type=str, help='test data directory')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for inference')
parser.add_argument('--dest_dir', default=SUBMISSIONS_DIR, type=str, help='Submissions directory')
parser.add_argument('--num_epochs', default=5, type=int, help='Number of training epochs')
parser.add_argument('--arch', default='CNN', type=str, help='Model architecture')


# predictions


def test(args):

    # create test data loader
    df = pd.read_csv(os.path.join(args.data_path, 'TrainV1.csv'))
    df = df.sample(n=3000, random_state=2020)
    test_labels = df.TARGET
    # create text files 
    os.makedirs(args.files_dest_directory, exist_ok=True)

    seq_to_text_file(
                    save_file_to=os.path.join(args.files_dest_directory, 'test_data.txt'),
                    sequences=df.SEQUENCE
                    )

    test_data_loader = get_data_loader(file=os.path.join(args.files_dest_directory, "test_data.txt"), 
                                    batch_size=args.batch_size,
                                    labels=test_labels, 
                                    task='test' )        

    res = []

    for num in range(args.n_folds):
        # create model
        model = create_model(show_summary=False)
        model.load_weights(os.path.join(args.ckpt_dir, f'enzyme_classifier-{num}.h5'))


        # make predictions
        results = model.evaluate(test_data_loader, steps=len(df) // args.batch_size) 

        res.append(results)

        del model
        del results

    
    res = np.array(res)

    # compute results 
    avg_loss = res[:, 0].mean()
    avg_acc = res[:, 1].mean()

    return avg_loss, avg_acc



        
    

def make_prediction(model, dataset:pd.DataFrame, batch_size=1):
    sequences = dataset.SEQUENCE
    try:
        # encode amino acid sequence
        encoded_seqs = [encode_sequence(sequence, label=None)[0] for sequence in sequences]
        d = tf.data.Dataset.from_generator(lambda : encoded_seqs, output_types=(tf.int64))
        d = d.padded_batch(batch_size, padded_shapes=([None]), drop_remainder=False)
        # predict corresponding class & get probability
        model.trainable = False
        logits = model.predict(d,  verbose=1)
        classes = [v.argmax() for v in logits]
        probas = [logits[i][c] for i, c in enumerate(classes)]    
    
    except:
        # encode amino acid sequence
        dataset['LABEL'] = 0
        test_labels = dataset['LABEL']
        seq_to_text_file(save_file_to=os.path.join(DATA_PATH, 'test', f'test_data.txt'), sequences=dataset.SEQUENCE)
        
        d = get_data_loader(file=os.path.join(DATA_PATH, 'test', f'test_data.txt'), 
                            batch_size=batch_size, 
                            labels=test_labels, 
                            task='test')
  
        # predict corresponding class & get probability
        model.trainable = False
        logits = model.predict(d,  verbose=1)
        classes = [v.argmax() for v in logits]
        probas = [logits[i][c] for i, c in enumerate(classes)]

    return classes, probas



def make_predictions(args):
    df = pd.read_csv(os.path.join(args.data_path, 'Test.csv'))

    preds = []
    probas = []
    model = create_model(show_summary=False, arch=args.arch)

    for num in range(args.n_folds):
        # create model
        model.load_weights(os.path.join(args.ckpt_dir, f'enzyme_classifier-{num}.h5'))

        # make prediction
        classes, probs = make_prediction(model=model, dataset=df, batch_size=args.batch_size)
        
        # save prediction to ensemble 
        preds.append(classes)
        probas.append(probs)


    
    return np.array(preds).T, np.array(probas).T



def get_ensemble_results(classes, probas):

    predicted_classes = []
    for line in tqdm(range(len(classes)), desc='Please wait models are voting'):
        pred = Counter(classes[line]).most_common(1)[0][0]
        predicted_classes.append("class"+str(pred))

    return predicted_classes


def save_submission_file(fn, df, args):
    dest_dir = args.dest_dir
    df[['SEQUENCE_ID', 'LABEL']].to_csv(os.path.join(dest_dir, fn), index=False)

    print(f'[INFO] submission file saved to {dest_dir} as {fn}.\nGood luck...fingers crossed !')


if __name__=='__main__':
    args = parser.parse_args()
    #df = pd.read_csv(os.path.join(args.data_path, 'Test.csv'))

    #s = df.SEQUENCE.tolist()[:5000]
    #model = create_model(show_summary=False)
    #model.load_weights(os.path.join(args.ckpt_dir, 'enzyme_classifier-1.h5'))
    classes, probas = make_predictions(args)

    final_predictions = get_ensemble_results(classes=classes, probas=probas)

    test_df = pd.read_csv(os.path.join(args.data_path, 'Test.csv'))
    test_df['LABEL'] = final_predictions

    print(test_df.head())
    fname = f'enz-{args.arch}-based-n_folds-{args.n_folds}-lr-{args.lr}-tbs-{args.train_bs}-valid_bs-{args.validation_bs}-n_epochs-{args.num_epochs}.csv'
    save_submission_file(fn=fname, df=test_df, args=args)
    
    #print(f'\n[INFO] Predicted class {c} with {pr:.3f}% confidence')
    #res = test(args)

    #print(f'[INFO] ')
    #print('')
    #print('*'*18)
    #print(f'Test Results')
    #print('*'*18)
    #print(f'> Avg Acc : \t{res[1]*100:.4f}% \t| Avg loss : {res[0]:.4f}')

    #predictions, probas = make_predictions(args)

    #print(predictions.shape, probas.shape)

    
