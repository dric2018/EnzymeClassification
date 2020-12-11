import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import transformers
import os
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datasets import EnzymeDataset
import argparse
import sys 


###### Constants
data_dir = os.path.join('../data/TrainV1.csv')

###### argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=data_dir,  type=str, help='data source directory')



class EnzymeClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, max_seq_len=512, n_layers=10, lr=5e-3,n_letters=24, n_classes=20, embed_size=512,device='cuda', *args, **kwargs):
        super(EnzymeClassifier, self).__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(self.hparams.n_letters, self.hparams.embed_size)

        self.lstm = nn.LSTM(self.hparams.embed_size, 
                            self.hparams.hidden_size,
                            self.hparams.n_layers,
                            batch_first=True, 
                            bidirectional=False)


        self.fc = nn.Linear(128, self.hparams.n_classes)


    def forward(self, x):
        h0 = torch.zeros(self.hparams.n_layers, x.size(0), self.hparams.hidden_size)
        c0 = torch.zeros(self.hparams.n_layers, x.size(0), self.hparams.hidden_size)

        embed = self.embedding(x.unsqueeze(0))
        print(embed.shape)
        packed = pack_padded_sequence(embed, [self.hparams.input_size], batch_first=False)
        print(packed)
        out, _ = self.lstm(packed, (h0.to(self.hparams.device), c0.to(self.hparams.device)))
        out = out.reshape(out.shape[0], -1)
        print("shape before fc",out.shape)

        return self.fc(out)


    def training_step(self, batch, batch_idx):
        xs, ys = batch['sequence'], batch['target']

        logits = self(xs)

        acc = self.get_acc(logits, ys)
        loss = self.get_loss(logits, ys)

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', acc, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss":loss, 'train_acc':acc, 'train_loss':loss}


    def validation_step(self, batch, batch_idx):
        xs, ys = batch['sequence'], batch['target']

        logits = self(xs)

        acc = self.get_acc(logits, ys)
        loss = self.get_loss(logits, ys)

        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss":loss, 'val_acc':acc, 'val_loss':loss}


    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

        return opt


    def test_step(self, batch, batch_idx):
        pass


    def get_loss(self, logits, targets):
        
        return nn.CrossEntropyLoss()(logits, targets)


    def get_acc(self, logits, targets):
        preds = logits.argmax(1)

        return (preds == targets).float().mean()




########## Utils 

# sample code from : https://github.com/pytorch/examples/blob/master/word_language_model/model.py

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)



class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)





if __name__=='__main__':
    args = parser.parse_args()
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 600
    train = pd.read_csv(args.data_path)
    INPUT_SIZE = 600
    ds = EnzymeDataset(df=train, task='train', max_seq_len=MAX_SEQ_LEN)
    dl = DataLoader(dataset=ds, 
                    batch_size=BATCH_SIZE,
                    num_workers=os.cpu_count())
    
    net = EnzymeClassifier(input_size=INPUT_SIZE, 
                            hidden_size=128, 
                            max_seq_len=MAX_SEQ_LEN,
                            n_layers=10,
                            n_classes=20)
    net.to('cuda')

    print(net)

    for data in dl:
        xs, ys = data['sequence'], data['target']
        print(xs.shape)
        out = net(xs.to('cuda'))
        print(out.shape)

        l = net.get_loss(out, ys.to('cuda'))
        acc = net.get_acc(out, ys.to('cuda'))

        print(f"loss = {l} acc = {acc}")

        break