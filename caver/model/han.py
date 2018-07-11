import torch
from torch import nn

from .base import BaseModule
from ..config import ConfigHAN
from ..utils import update_config


class HAN(BaseModule):
    """
    :param hidden_dim: dimension of hidden layer
    :type hidden_dim: int
    :param layer_num: number of hidden layer
    :type layer_num: int
    :param bidirectional: use bidirectional lstm layer?
    :type bidirectional: bool

    This model is the implementation of HAN(only word encoder and word attention)
    from `han_paper`_:
    Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.

    .. _han_paper: http://www.aclweb.org/anthology/N16-1174

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.config = update_config(ConfigHAN(), **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_dim
        )
        self.rnn = nn.GRU(
            self.config.embedding_dim,
            self.config.hidden_dim,
            self.config.layer_num,
            batch_first=True,
            bidirectional=self.config.bidirectional,
        )
        self.attention = Attention(self.config)
        self.mlp = nn.Linear(self.config.hidden_dim * (2 if self.config.bidirectional else 1), self.config.label_num)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return torch.zeros(
                self.config.layer_num * (2 if self.config.bidirectional else 1),
                batch_size, self.config.hidden_dim
            ).to(self.device)

    def forward(self, input_data):
        batch_size = input_data.size(0)
        hidden = self.init_hidden(batch_size)
        embedded = self.embedding(input_data)
        context, hidden = self.rnn(embedded, hidden)
        context = self.attention(context)
        return self.sigmoid(self.mlp(context))


class Attention(nn.Module):
    """
    Attention layer of HAN.
    """
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.linear = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.uw = torch.randn(self.config.hidden_dim, 1, requires_grad=True).to(self.device)

    def forward(self, context):
        batch_size = context.size(0)
        u = self.tanh(self.linear(context))
        weight = self.softmax(u.bmm(self.uw.unsqueeze(0).repeat(batch_size, 1, 1)))
        return weight.transpose(1, 2).bmm(context).view(batch_size, -1)
