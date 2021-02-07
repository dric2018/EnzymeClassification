########################################
# Instadeep Enzyme Classification
# by : Cedric MANOUAN A.K.A I_Am_Zeus_AI
########################################

import os
import numpy as np 
import pandas as pd
from tqdm import tqdm 
import swifter
import argparse
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings(action='ignore')


###### Constants
data_dir = os.path.join('../data')

###### argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=data_dir,  type=str, help='data source directory')




####### Utilities finctions

def calc_max_len(df:pd.DataFrame):
    maxi_len = 0
    for row in df.iterrows():
        if len(row.SEQUENCE) > maxi_len:
            maxi_len = len(row.SEQUENCE)
    return maxi_len


def calc_min_len(df:pd.DataFrame):
    mini_len = np.inf
    for row in df.iterrows():
        if len(row.SEQUENCE) < mini_len:
            mini_len = len(row.SEQUENCE)
    return mini_len


def label_to_int(label, class_dict):
    return class_dict[label]

def get_len(row):
    return len(row.SEQUENCE)

def save_dataframe(df, dest, filename, save_index=False):
    try:
        dest_path = os.path.join(dest, filename)
        df.to_csv(dest_path, index=save_index)
        message = f"Dataframe saved to {dest_path}"

    except Exception as ex:
        message = f"{ex}"

    print(message)



if __name__ == '__main__':

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    class_dict = {label:idx for idx, label in enumerate(sorted(df.LABEL.unique().tolist()))}

    df['TARGET'] = df.swifter.progress_bar(enable=True, desc='Computing sequence lengths').apply(lambda row: label_to_int(label=row.LABEL, class_dict=class_dict), axis=1)
    df['LENGTH'] = df.swifter.progress_bar(enable=True, desc='Computing sequence lengths').apply(lambda row: get_len(row), axis=1)

    print(df.head())

    # save dataframe
    save_dataframe(df=df, dest=os.path.dirname(args.data_path), filename='TrainV1.csv', save_index=False)
