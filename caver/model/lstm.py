#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from .base import BaseModule
from ..config import ConfigLSTM
from ..utils import update_config


class LSTM(BaseModule):
    """
    :param hidden_dim: hidden layer dimension
    :type hidden_dim: int
    :param layer_num: num of hidden layer
    :type layer_num: int
    :param bidirectional: use bidirectional lstm layer?
    :type bidirectional: bool

    Simple LSTM model

    text -> embedding -> lstm -> mlp -> sigmoid

    """
    def __init__(self, hidden_dim=100, embedding_dim=100, vocab_size=1000,
                 label_num=100, device="cpu", layer_num=2, dropout=0.3,
                 batch_first=True, bidirectional=True):
        super().__init__()
        # self.config = update_config(ConfigLSTM(), **kwargs)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## starts with _ stands for static attr, otherwise nn layers
        self._layer_num = layer_num
        self._bidirectional = bidirectional
        self._device = device
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._label_num = label_num
        self._dropout = dropout
        self._batch_first = batch_first

        self.embedding = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.dropout = nn.Dropout(self._dropout)
        self.lstm = nn.LSTM(self._embedding_dim,
                            self._hidden_dim,
                            self._layer_num,
                            batch_first=self._batch_first,
                            bidirectional=self._bidirectional,
                            dropout=self._dropout)
        self.predictor = nn.Linear(self._hidden_dim*2 if self._bidirectional else self._hidden_dim*1,
                                   self._label_num)


    def init_hidden(self, batch_size):
        return (
            torch.zeros(
                self.layer_num * (2 if self.bidirectional else 1),
                batch_size, self.hidden_dim
            ).to(self.device),

            torch.zeros(
                self.layer_num * (2 if self.bidirectional else 1),
                batch_size, self.hidden_dim
            ).to(self.device)
        )

    def forward(self, sequence):
        #### sentence = [batch_size , sent len]

        # batch_size = sequence.size(0)
        # hidden = self.init_hidden(batch_size)
        embedded = self.embedding(sequence)
        #### embedded = [batch_size , sent len , embedding dim]
        embedded = self.dropout(embedded)

        self.lstm.flatten_parameters()
        output, (hidden, cell) = self.lstm(embedded)
        #### output = [batch_size, sent len, hidden_dim x num_directions]
        #### hidden = [batch size, num layers x num directions, hiddim dim]
        #### cell = [batch size, num layers x num directions, hiddim dim]

        output_feature = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output_feature = self.dropout(output_feature)
        #### output_feature = [batch_size, hidden_dim x num_directions]

        preds = self.predictor(output_feature.squeeze(0))
        return preds
