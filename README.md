# Enzyme Classification
Here I am dealing with a bioinformatics problem known as enzyme family classification for a challenge organized by Instadeep and Zindi africa.

Given an amino acid sequence, you have to predict which of 20 classes it is from. As you can see it is an `Sequence classification problem`. 

# Usage 

## Hardware specifications
* Gaming laptop or any computer (e.g workstation) with at least:
>> 1 NVIDIA GPU (VRAM >= 6 GB since I used a GTX 1060 Max-Q)

>> 4 Cores CPU (>= Intent core i5 )

>> 16 GB of RAM

>> 50 GB of free space 

## Training  
1 - Tensorflow version 
* Create `FOLD` column in `Train.csv` file for cross-validation purpose
 ```
$ python utils.py --n_fols 5 --data_path <path_to_data_directory>
```

* Run `train.py` file with your arguments according to your experiements and your hardware specifications
```
$ python train.py --num_epochs 5 --lr 1e-3 --train_bs 256 --validation_bs 128 --n_folds 5
```
2- Pytorch version

## Inference

1 - Load trained weights from [here](#)

2 - Run inference script
```
$ python inference.py --batch_size 256
```


# Disclaimer
Some of the utilities used in this repository are based on the [ZINDI UMOJAHACK TUNISIA 2020](https://zindi.africa/hackathons/umojahack-tunisia/data) starter notebook. They have been used as a code base for the experiments and gave a very stable understanding on data preprocessing for this task.