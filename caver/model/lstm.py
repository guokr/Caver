import torch
from torch import nn

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
    
    Simpole LSTM model

    text -> embedding -> lstm -> mlp -> sigmoid

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.config = update_config(ConfigLSTM(), **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_dim
        )

        self.lstm = nn.LSTM(
            self.config.embedding_dim,
            self.config.hidden_dim,
            self.config.layer_num,
            batch_first=True,
            bidirectional=self.config.bidirectional,
        )
        self.mlp = nn.Linear(self.config.hidden_dim, self.config.label_num)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return (
            torch.zeros(
                self.config.layer_num * (2 if self.config.bidirectional else 1),
                batch_size, self.config.hidden_dim
            ).to(self.device),
            torch.zeros(
                self.config.layer_num * (2 if self.config.bidirectional else 1),
                batch_size, self.config.hidden_dim
            ).to(self.device)
        )

    def forward(self, input_data):
        batch_size = input_data.size(0)
        hidden = self.init_hidden(batch_size)
        embedded = self.embedding(input_data)
        _, hidden = self.lstm(embedded, hidden)
        label = self.mlp(hidden[0].view(batch_size, -1))
        return self.sigmoid(label)
