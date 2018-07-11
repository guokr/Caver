import os
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim

from . import model
from .data import TextData
from .utils import make_batches, init_weight, zero_padding, transform2onehot
from .utils import get_top_label_with_logits, update_config, recall_at_k
from .utils import load_embedding
from .config import Config


class Trainer:
    """
    :param model_name: name of model, case sensitive.
    :type model_name: str
    :param data_path: file path of data
    :type data_path: str

    You can pass your own config as parameters to replace default value in
    :class:`caver.config.Config`.

    GPU will be used if available.
    """
    def __init__(self, model_name, data_path, **kwargs):
        self.config = update_config(Config(), **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = TextData(data_path)

        assert hasattr(model, model_name)
        self.name = model_name
        self.model = getattr(model, model_name)(
            vocab_size=len(self.data.word2index),
            label_num=len(self.data.label2index)
        ).to(self.device)
        self.model.apply(init_weight)

        print(model)

        self.loss_func = self.config.loss_func \
            if isinstance(self.config.loss_func, torch.nn.Module) \
            else nn.BCELoss()
        self.optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

    def init_model_embedding(self):
        """
        Init embedding layer use pre-trained model.

        This will be used if :class:`caver.config.Config.embedding_file` is not None.
        """
        print('Init embedding layer use pre-trained file')
        embedding_weight = load_embedding(
            self.config.embedding_file,
            self.config.embedding_dim,
            len(self.index2word),
            self.index2word
        )
        embedding_weight = torch.from_numpy(embedding_weight).type(torch.FloatTensor)
        self.model.embedding.weight = nn.Parameter(embedding_weight)
        if not self.config.embedding_train:
            self.model.embedding.weight.requires_grad = False

        self.model.to(self.device)

    def train(self):
        x, y = self.data.prepare()
        self.index2label = dict([(a, b) for (b, a) in self.data.label2index.items()])
        self.index2word = dict([(a, b) for (b, a) in self.data.word2index.items()])

        if self.config.embedding_file:
            self.init_model_embedding()

        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=self.config.valid)

        for i in range(self.config.epoches):
            print('Running epoch {}/{}'.format(i + 1, self.config.epoches))
            self.train_step(train_x, train_y, val_x, val_y)
            self.model.save(os.path.join(self.config.save_path, self.name + '_{}.pth'.format(i + 1)))


    def train_step(self, train_x, train_y, val_x, val_y):
        batches = make_batches(len(train_x), self.config.batch_size)
        total_loss = 0.0

        for i, (start, end) in enumerate(batches):
            input_data = torch.from_numpy(
                zero_padding(train_x[start:end], self.config.sentence_length)
            ).type(torch.long).to(self.device)
            input_label = torch.from_numpy(
                transform2onehot(train_y[start:end], len(self.index2label))
            ).type(torch.float).to(self.device)

            self.model.train()
            self.model.zero_grad()
            logits = self.model(input_data)
            loss = self.loss_func(logits, input_label)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if i % (self.config.valid_interval - 1) == 0:
                self.eval_step(val_x, val_y)

        self.eval_step(val_x, val_y)


    def eval_step(self, x, y):
        batches = make_batches(len(x), self.config.batch_size)
        self.model.eval()
        total_loss, total_recall = 0.0, 0.0
        total_preds = []

        with torch.no_grad():
            for start, end in batches:
                input_data = torch.from_numpy(
                    zero_padding(x[start:end], self.config.sentence_length)
                ).type(torch.long).to(self.device)
                input_label = torch.from_numpy(
                    transform2onehot(y[start:end], len(self.index2label))
                ).type(torch.float).to(self.device)

                self.model.eval()
                logits = self.model(input_data)
                loss = self.loss_func(logits, input_label).item()
                total_loss += loss
                preds = [get_top_label_with_logits(p, self.index2label, self.config.recall_k) for p in logits.cpu().numpy()]
                total_preds.extend(preds)

        total_labels = [[self.index2label.get(i) for i in item] for item in y]
        total_text = [' '.join([self.index2word.get(i, '') for i in item]) for item in x]
        with open('{}_eval.txt'.format(self.name), 'w', encoding='utf-8') as f:
            for text, pred, label in zip(total_text, total_preds, total_labels):
                recall = recall_at_k(pred, label, self.config.recall_k)
                total_recall += recall
                f.write('{}\nLabel: {}\nPred: {}\n'.format(text, label, pred))
                f.write('Recall@{}: {}\n'.format(self.config.recall_k, recall))
                f.write('-' * 80 + '\n')

        print('[+] Total loss: {}\nAverage recall@{}: {}'.format(
            total_loss, self.config.recall_k, total_recall / len(y))
        )
        print('-' * 80 + '\n')
