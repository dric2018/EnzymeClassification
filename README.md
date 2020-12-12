# Enzyme Classification
Here I am dealing with a bioinformatics problem known as enzyme family classification for a challenge organized by Instadeep and Zindi africa.

Given an amino acid sequence, you have to predict which of 20 classes it is from. As you can see it is an `Sequence classification problem`. 

# Usage 

## Training  
1 - Tensorflow version 
* Create `FOLD` column in `Train.csv` file for cross-validation purpose
 ```
$ python utils.py --n_fols 5 --data_path <path_to_data_directory>
```

* Run `train.py` file with your arguments according to your experiements and your hardware specifications
```
$ python train.py --num_epochs 3 --lr 1e-3 --train_bs 512 --validation_bs 256 --n_folds [same_as_train_csv_n_splits] 
```
2- Pytorch version

## Inference

1 - Load trained weights from [here](#)

2 - Run inference script
```
$ python inference.py --batch_size 64
```


# Disclaimer
Some of the utilities used in this repository are based on the [ZINDI UMOJAHACK TUNISIA 2020](https://zindi.africa/hackathons/umojahack-tunisia/data) starter notebook. They have been used as a code base for the experiments and gave a very stable understanding on data preprocessing for this task.