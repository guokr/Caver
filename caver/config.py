class Config:
    """
    Basic config. All model config should inherit this.
    """
    # #: index dir of word and label.
    # index_path = os.path.join(os.path.abspath(os.path.curdir), 'caver_index')
    # word2index = os.path.join(index_path, 'word2index.json')
    # label2index = os.path.join(index_path, 'label2index.json')
    #: embedding dimension
    embedding_dim = 256
    # sentence_length = 64
    # #: min word count, word frequence below this will be ignored
    # min_word_count = 5
    #: min label count, label frequence below this will be ignored
    # min_label_count = 100
    # #: validatoin size
    # valid = 0.15
    #: batch size
    batch_size = 256
    #: epoch num for train
    epoch = 10
    # #: interval of validataion
    # valid_interval = 200
    #: recall@k
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
    #: data directory
    input_data_dir = "dataset"
    #: train filename
    train_filename = "nlpcc_train.tsv"
    #: validation filename
    valid_filename = "nlpcc_valid.tsv"
    #: save processed data directory
    output_data_dir = "processed_data"
    #: checkpoint directory
    checkpoint_dir = "checkpoints"
    #: gpu device number
    master_device = 0
    #: use multi gpu or not
    multi_gpu = False
    #: learning rate
    lr = 1e-4


class ConfigCNN(Config):
    """
        CNN model config.
    """
    #: model name
    model = 'CNN'
    #: filter number
    filter_num = 6
    #: list of filter size
    filter_sizes = [2, 3, 4]


class ConfigSWEN(Config):
    window = 3
    embedding_drop = 0.2


class ConfigLSTM(Config):
    """
        LSTM model config.
    """
    #: model name
    model = 'LSTM'
    #: hidden number
    hidden_dim = 128
    #: hidden layer number
    layer_num = 1
    #: use bidirectional LSTM or not
    bidirectional = False


class ConfigHAN(Config):
    hidden_dim = 64
    layer_num = 1
    bidirectional = True


class ConfigfastText(Config):
    """
        fastText model config.
    """
    #: model name
    model = 'fastText'
