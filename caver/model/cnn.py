#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModule


class CNN(BaseModule):
    """
    :param filter_num: number of filter
    :type filter_num: int
    :param filter_size: size of filter
    :type filter_size: list

    This is the implementation of CNN from `cnn_paper`_:
    Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).

    .. _cnn_paper: https://arxiv.org/pdf/1408.5882.pdf

    text -> embedding -> conv -> relu -> BatchNorm -> max_pool -> mlp -> sigmoid
    """
    def __init__(self, vocab_size=1000, embedding_dim=100,
                 filter_num=100, filter_sizes=[2,3,4],
                 label_num=100, dropout=0.3):
        super().__init__()
        # self.config = update_config(ConfigCNN(), **kwargs)
        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._filter_sizes = filter_sizes
        self._dropout = dropout
        self._filter_num = filter_num
        self._label_num = label_num

        # need or not
        self._hidden_dim = len(self._filter_sizes) * self._filter_num

        self.embedding = nn.Embedding(self._vocab_size,
                                      self._embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=self._filter_num,
                                              kernel_size=(filter_size, self._embedding_dim)
                                              )
                                    for filter_size in self._filter_sizes])

        self.dropout = nn.Dropout(self._dropout)

        #?? hidden or not should be test
        self.bn = nn.BatchNorm1d(self._hidden_dim)
        self.predictor = nn.Linear(self._hidden_dim,
                                   self._label_num)


    def forward(self, sequence):
        # print(sequence.shape)
        embedded = self.embedding(sequence)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = self.dropout(self.bn(torch.cat(pooled, dim=1)))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.predictor(cat)


    def predict(self, batch_sequence_text, vocab_dict, device="cpu", top_k=5):
        batch_preds = self._predict_text(batch_sequence_text=batch_sequence_text,
                                         vocab_dict=vocab_dict,
                                         device=device,
                                         top_k=top_k)

        batch_top_k_value, batch_top_k_index = torch.topk(torch.sigmoid(batch_preds), k=top_k, dim=1)

        return batch_top_k_index


    def _predict_text(self, batch_sequence_text, vocab_dict, device="cpu", top_k=5):
        """
        do prediction for tokenized text in batch way

        CNN is special in processing <pad>

        vocab_dict: {"word": 1, "<pad>": 0}
        """
#            # print(type(vocab_dict))
#            # sequence_text: "我 喜欢 吃 苹果"
#            tokenized = sequence_text.split()
#            # at least the CNN's filter sizes' longest one
#            if len(tokenized) < max(self._filter_sizes):
#                tokenized += ["<pad>"] * (max(self._filter_sizes) - len(tokenized))
#            indexed = [vocab_dict[t] for t in tokenized]
#            indexed = torch.LongTensor(indexed).to(device)
#            indexed = indexed.unsqueeze(0) # set batch_size to 1
#            preds = self.forward(indexed)
#            return preds

        batch_tokenized = [seq.split() for seq in batch_sequence_text]
        # print(batch_tokenized)
        batch_longest = max(map(len, batch_tokenized))
        batch_padding_threshold = max(max(self._filter_sizes), batch_longest)
        # print(batch_longest)
        for sample in batch_tokenized:
            if len(sample) < batch_padding_threshold:
                sample  += ["<pad>"] * (batch_padding_threshold - len(sample))

        # print(batch_tokenized)
        batch_indexed =[[vocab_dict[sample_token] for sample_token in sample] for sample in batch_tokenized]
        # print(batch_indexed)
        indexed = torch.LongTensor(batch_indexed).to(device)
        # print(indexed)
        batch_preds = self.forward(indexed)
        return batch_preds
