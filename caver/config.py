import os

class Config:
    """
    Basic config. All model config should inherit this.
    """
    # #: index dir of word and label.
    # index_path = os.path.join(os.path.abspath(os.path.curdir), 'caver_index')
    # word2index = os.path.join(index_path, 'word2index.json')
    # label2index = os.path.join(index_path, 'label2index.json')
    embedding_dim = 256
    # sentence_length = 64
    # #: min word count, word frequence below this will be ignored
    # min_word_count = 5
    #: min label count, label frequence below this will be ignored
    # min_label_count = 100
    # #: validatoin size
    # valid = 0.15
    batch_size = 256
    epoch = 10
    # #: interval of validataion
    # valid_interval = 200
    #: racall@k
    recall_k = 5
    # #: segment model, you can choose `jieba` or `pyltp`, if not set, `plane.segment`
    # #: will be uesd.
    # cut_model = None
    # #: model will be saved in this dir
    # save_path = 'checkpoint'
    # vocab_size = None
    # label_num = None
    # #: pre-trained embedding file, this will be used to init embedding layer if offered.
    # embedding_file = None
    # #: train the embedding layer when training the model
    # embedding_train = True
    # loss_func = None
    # optimizer = None
    #: dropout rate
    dropout = 0.15
    input_data_dir = "dataset"
    train_filename = "nlpcc_train.tsv"
    valid_filename = "nlpcc_valid.tsv"
    output_data_dir = "processed_data"
    checkpoint_dir = "checkpoints"
    master_device = 0
    multi_gpu = False
    lr = 1e-4


class ConfigCNN(Config):
    model = 'CNN'
    filter_num = 64
    filter_sizes = [2, 3, 4]


class ConfigSWEN(Config):
    window = 3
    embedding_drop = 0.2


class ConfigLSTM(Config):
    model = 'LSTM'
    hidden_dim = 128
    layer_num = 1
    bidirectional = False


class ConfigHAN(Config):
    hidden_dim = 64
    layer_num = 1
    bidirectional = True


class ConfigfastText(Config):
    model = 'fastText'
