import os


class Config:
    data_dir = os.path.abspath('../data/')
    models_dir = os.path.abspath('../models')
    logs_dir = os.path.abspath('../logs')
    num_epochs = 50
    lr = 3e-2
    base_model = 'GRU'
    weight_decay = .01
    eps = 1e-8
    max_seq_len = 512
    train_batch_size = 512
    test_batch_size = 256
    seed_val = 2021
    num_workers = 4
    num_classes = 20
    default_vocab = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'U': 18,'V': 19,'W': 20,'X': 21,'Y': 22,'Z': 23, '<PAD>':24}