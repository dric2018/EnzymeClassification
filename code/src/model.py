# IMPPORTS
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from .config import Config


# CONSTANTS DEFINITION


class Model(pl.LightningModule):
    def __init__(self, config: dict, 
                 input_size=512,
                 sequence_len=512,
                 hidden_size=300, 
                 dropout_prob=.2,
                 embedding_dim=350,
                 num_layers=3):
        super(Model, self).__init__()

        try:
            self.save_hyperparameters()
        except:
            pass
        # self.embedding = nn.Embedding(
        #    num_embeddings = len(Config.default_vocab), 
        #    embedding_dim=self.hparams.hidden_size)
        
        self.embedding = nn.Embedding(
                num_embeddings=len(Config.default_vocab)+1, 
                embedding_dim = self.hparams.embedding_dim
            )
        self.encoder = nn.GRU(
                input_size=self.hparams.embedding_dim, 
                hidden_size=self.hparams.hidden_size, 
                num_layers=self.hparams.num_layers, 
                batch_first=True, 
                dropout=self.hparams.dropout_prob, 
                bidirectional=True
            )
        
         
        # create classification layer
        
        self.classifier = nn.Linear(
            in_features=2*self.hparams.hidden_size, 
            out_features=Config.num_classes
        )
        self.dropout = nn.Dropout(p=self.hparams.dropout_prob)

    def forward(self, seq):
        """
        inspired from : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
        """
        # compute embedding
        out = self.embedding(seq)
        #compute init hidden size
        h0 = th.zeros(
             self.hparams.num_layers * 2, # if bidirectional multiply num_layers by 2
             seq.size(0),
             self.hparams.hidden_size).to(self.device)
        
        out, _ = self.encoder(out, h0)
        out = out.reshape(out.shape[0], -1)
        
        print(out.shape)
        # out = self.classifier(out)
        
        return out

    def training_step(self, batch, batch_idx):
        input_ids, y = batch['input_ids'], batch['trg']
        # forward pass
        logits = self(input_ids)
        preds = nn.LogSoftmax(dim=1)(logits)
        
        # compute metrics
        loss = th.nn.NLLLoss(ignore_index=Config.default_vocab["<PAD>"])(preds, y)
        acc = accuracy(preds.cpu(), y.cpu())

        self.log('train_acc', 
                 acc, 
                 prog_bar=True,
                 on_step=True, 
                 on_epoch=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": preds,
                'targets': y
                }

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        input_ids, y = batch['input_ids'], batch['trg']

        # forward pass
        logits = self(input_ids)
        preds = nn.LogSoftmax(dim=1)(logits)
            
        # compute metrics
        val_loss = th.nn.NLLLoss(ignore_index=Config.default_vocab["<PAD>"])(preds, y)
        val_acc = accuracy(preds.cpu(), y.cpu())

        self.log('val_loss', 
                 val_loss, 
                 prog_bar=True,
                 on_step=False, 
                 on_epoch=True
                )
        
        self.log('val_acc', 
                 val_acc, 
                 prog_bar=True,
                 on_step=False, 
                 on_epoch=True
                )

        return {'loss': val_loss,
                'accuracy': val_acc,
                "predictions": preds,
                'targets': y
                }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def configure_optimizers(self):
        opt = th.optim.AdamW(
            lr=Config.lr,
            params=self.parameters(),
            eps=Config.eps,
            weight_decay=Config.weight_decay
        )
        
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='max',
            factor=0.1,
            patience=3,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
            )
        return {"optimizer": opt, 
                "lr_scheduler": scheduler, 
                "monitor": "val_acc"}

