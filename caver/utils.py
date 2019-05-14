import numpy as np
from torch import nn
import torch
import os


def init_weight(layer):
    """
    Init layer weights and bias
    """
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(layer.weight.data, gain=np.sqrt(2))
        nn.init.constant_(layer.bias.data, 0.1)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(layer.weight.data, 1.0)
        nn.init.constant_(layer.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.eye_(layer.weight.data)
    # else:
    #     nn.init.normal_(layer.weight.data, 0, 0.1)


def update_config(config, **kwargs):
    """
    Update config attributes with key-value in kwargs.

    Keys not in config will be ignored.
    """
    for key in kwargs:
        if not hasattr(config, key):
            print('Ignore unknown attribute {}.'.format(key))
        else:
            setattr(config, key, kwargs[key])
            print('Attribute {} has been updated.'.format(key))

    return config


def zero_padding(x, length):
    result = np.zeros((len(x), length))
    for i, row in enumerate(x):
        for j, val in enumerate(row):
            if j >= length:
                break
            result[i][j] = val
    return result


def transform2onehot(y, num_class):
    label = np.zeros((len(y), num_class))
    for i, index in enumerate(y):
        for j in index:
            label[i][j] = 1
    return label


def scaler(x, minimal=0, maximal=1):
    std = (x - np.min(x)) / (np.max(x) - np.min(x))
    return std * (maximal - minimal) + minimal


def get_top_label_with_logits(logits, index2label, top=5):
    index = np.argsort(logits)[-top:]
    index = index[::-1]
    return [index2label.get(i, '<WRONG>') for i in index]


def recall_at_k(pred, y, k=5):
    if len(pred) > k:
        pred = pred[:k]

    hits = 0.0
    for p in pred:
        if p in y:
            hits += 1.0

    return hits / min(len(y), k)


def make_batches(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(num_batches)]


def load_embedding(embedding_file, dim, vocab_size, index2word):
    """
    :param embedding_file: path of embedding file
    :type embedding_file: str
    :param dim: dimension of vector
    :type dim: int
    :param vocab_size: size of vocabulary
    :type vocab_size: int
    :param index2word: index => word
    :type index2word: dict

    Load pre-trained embedding file.

    First line of file should be the number of words and dimension of vector.
    Then each line is combined of word and vectors separated by space.

    ::

        1024, 64 # 1024 words and 64-d
        a 0.223 0.566 ......
        b 0.754 0.231 ......
        ......

    """
    word2vec = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        print('Embedding file header: {}'.format(f.readline())) # ignore header
        for line in f.readlines():
            items = line.strip().split(' ')
            word2vec[items[0]] = [float(vec) for vec in items[1:]]

    embedding = [[]] * vocab_size
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    count_exist, count_not_exist = 0, 0
    for i in range(vocab_size):
        word = index2word[i]
        try:
            embedding[i] = word2vec[word]
            count_exist += 1
        except:
            embedding[i] = np.random.uniform(-bound, bound, dim)
            count_not_exist += 1

    print('word exists embedding:', count_exist, '\tword not exists:', count_not_exist)
    embedding = np.array(embedding)
    return embedding


def check_ensemble_args(args):
    status = True
    if not (
        os.path.exists(args.cnn)
        or os.path.exists(args.lstm)
        or os.path.exists(args.fasttext)
    ):
        status = False
        print("|ERROR| no model directory is exist")

    models = list(
        filter(
            lambda x: os.path.exists(x) == True, [args.cnn, args.lstm, args.fasttext]
        )
    )
    if len(models) < 2:
        status = False
        print("|ERROR| numbers of model to ensemble shouldn`t less than two")

    if len(args.model_ratio) > 0:
        if len(models) != len(args.model_ratio):
            status = False
            print("|ERROR| model ratio numbers not equal to model numbers ")
        elif sum(args.model_ratio.values()) != 1:
            status = False
            print("|ERROR| add all model`s ratio not equal one")

    if len(args.sentences) == 0:
        status = False
        print("|ERROR| sentences list can`t be empty")

    return status


def show_ensemble_args(args):
    dict_args = vars(args)
    print("=============== Command Line Tools Args ===============")
    for arg, value in dict_args.items():
        if isinstance(value, dict) and len(value) > 0:
            value = " ".join("{}_{}".format(k, v) for k, v in value.items())
        elif isinstance(value, dict) and len(value) == 0:
            continue
        elif isinstance(value, list):
            value = "'" + ",'".join(value) + "'"
        elif isinstance(value, str) and value == "":
            continue
        print("{:>20} <===> {:<20}".format(arg, value))
    print("=======================================================")


def set_config(config, args_dict):
    """
    Update config attributes with key-value in kwargs.

    Keys not in config will be ignored.
    """
    for key, value in args_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


class MiniBatchWrapper(object):
    """
    wrap the simple torchtext iter with multiple y label
    """
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            if self.y_vars is  not None:
                temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                y = torch.cat(temp, dim=1).float()
            else:
                y = torch.zeros((1))
            yield (x, y)

    def __len__(self):
        return len(self.dl)



