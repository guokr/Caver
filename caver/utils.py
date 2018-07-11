import numpy as np
from torch import nn

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

