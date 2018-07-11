import torch
from torch import nn

from .base import BaseModule
from ..config import ConfigSWEN
from ..utils import update_config


class SWEN(BaseModule):
    """
    :param window: avg_pool window
    :type window: int

    This model is the implementation of SWEN-hier from `swen_paper`_:
    Shen, Dinghan, et al. "Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms."

    .. _swen_paper: https://arxiv.org/abs/1805.09843

    text -> embedding -> avg_pool -> max_pool -> mlp -> sigmoid
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.config = update_config(ConfigSWEN(), **kwargs)

        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.embedding_dim, self.config.sentence_length
        )

        self.embedding_dropout = nn.Dropout(self.config.embedding_drop)
        self.avg_pool = nn.AvgPool1d(self.config.window)
        self.max_pool = nn.MaxPool1d(
            (self.config.sentence_length - self.config.window) // 3 - 1
        )
        self.dropout = nn.Dropout(self.config.drop)
        self.mlp = nn.Linear(self.config.embedding_dim, self.config.label_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        embedded = self.embedding(input_data).transpose(1, 2)
        hidden = self.embedding_dropout(embedded)
        hidden = self.avg_pool(hidden)
        # hidden = self.avg_pool(embedded)
        hidden = self.max_pool(hidden)
        hidden = hidden.view(-1, self.config.embedding_dim)
        return self.sigmoid(self.mlp(self.dropout(hidden)))
