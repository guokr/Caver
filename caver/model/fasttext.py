#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModule


class FastText(BaseModule):
    """
    :param config: ConfigfastText which contains fastText configures

    Original FastText re-implementaion
    """

    def __init__(self, config, vocab_size=1000, label_num=100):
        super().__init__()
        self._vocab_size = vocab_size
        self._embedding_dim = config.embedding_dim
        self._label_num = label_num

        self.embedding = nn.Embedding(self._vocab_size, self._embedding_dim)
        self.predictor = nn.Linear(self._embedding_dim, self._label_num)

    def forward(self, sentence):
        #        #### sentence = [batch_size, sent length]
        embedded = self.embedding(sentence)
        #        #### embedded = [batch size, sent length, embedding dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        #        #### before squeeze: [batch_size, 1, embedding_dim]
        #        #### after squeeze: [batch_size, embedding_dim]
        preds = self.predictor(pooled)

        return preds
