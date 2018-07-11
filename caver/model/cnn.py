import torch
from torch import nn

from .base import BaseModule
from ..config import ConfigCNN
from ..utils import update_config

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
    def __init__(self, **kwargs):
        super().__init__()
        self.config = update_config(ConfigCNN(), **kwargs)

        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.embedding_dim, self.config.sentence_length
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ModuleList([
            self.conv_relu_bn_pool(
                self.config.embedding_dim, self.config.filter_num, size, self.config.sentence_length
            ) for size in self.config.filter_size
        ])
        self.hidden_dim = len(self.config.filter_size) * self.config.filter_num
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(self.config.drop)
        self.mlp = nn.Linear(self.hidden_dim, self.config.label_num)
        self.sigmoid = nn.Sigmoid()

    def conv_relu_bn_pool(self, in_channel, out_channel, kernel_size, dim):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size),
            self.relu,
            nn.BatchNorm1d(out_channel),
            nn.MaxPool1d(dim - kernel_size + 1),
        )

    def forward(self, input_data):
        embedded = self.embedding(input_data).transpose(1, 2)
        hidden = torch.cat([
            conv(embedded) for conv in self.conv
        ], 2)
        hidden = self.bn(hidden.view(-1, self.hidden_dim))
        hidden = self.mlp(self.dropout(hidden))
        return self.sigmoid(hidden)
