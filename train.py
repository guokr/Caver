#!/usr/bin/env python
# encoding: utf-8

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import dill as pickle

from torchtext.data import Field, TabularDataset, BucketIterator
from caver.model import LSTM, FastText, CNN
from caver.utils import MiniBatchWrapper
from caver.trainer import Trainer
from caver.config import *
import caver.utils as utils


parser = argparse.ArgumentParser(description="Caver training")
parser.add_argument("--model", type=str, choices=["fastText", "LSTM", "CNN"],
                    help="choose the model", default="CNN")
parser.add_argument("--input_data_dir", type=str, help="data dir", default='dataset')
parser.add_argument("--train_filename", type=str, default="nlpcc_train.tsv")
parser.add_argument("--valid_filename", type=str, default="nlpcc_valid.tsv")
parser.add_argument("--epoch", type=int, help="number of epoches", default=10)
parser.add_argument("--batch_size", type=int, default=16,
                    help="batch size for each GPU card")
parser.add_argument("--output_data_dir", type=str, default="processed_data")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="dir for checkpoints saving")
parser.add_argument("--master_device", type=int, default=0)
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument("--lr", type=float, help="initial learning rate", default=1e-4)

# model normal parameters
parser.add_argument("--embedding_dim", type=int, default=256, help="embedding layer dimension")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout ratio")

# LSTM model parameters
parser.add_argument("--layer_num", type=int, default=2, help="layer number")
parser.add_argument("--bidirectional", type=bool, default=True, help="bidirectional lstm or not")
parser.add_argument("--hidden_dim", type=int, default=100, help="hidden layer dimension")

# CNN model parameters
parser.add_argument("--filter_num", type=int, default=100, help="filter number")
parser.add_argument("--filter_sizes", type=list, default=[2,3,4], help="filter sizes")


args = parser.parse_args()

'''
set corresponding parameters to the corresponding model
'''
def update_args(args):
    if args.model == "LSTM":
        config = utils.set_config(ConfigLSTM(), vars(args))
    elif args.model == "CNN":
        config = utils.set_config(ConfigCNN(), vars(args))
    elif args.model == "fastText":
        config = utils.set_config(ConfigfastText(), vars(args))
    return config

'''
check base config is valid or not
'''
def check_args(config):
    status = True
    if not os.path.exists(os.path.join(config.input_data_dir, config.train_filename)):
        status = False
        print("|ERROR| train file doesn't exist")

    if not os.path.exists(os.path.join(config.input_data_dir, config.valid_filename)):
        status = False
        print("|ERROR| valid file doesn't exist")

    if torch.cuda.is_available() == False:
        status = False
        print("|ERROR| Currently we dont support CPU training")

    if torch.cuda.device_count() == 1 and config.multi_gpu == True:
        status = False
        print("|ERROR| We only detected {} GPU".format(torch.cuda.device_count()))

    if os.path.isdir(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) != 0:
        status = False
        # exist but not empty
        print("|ERROR| save dir must be empty")

    if not os.path.isdir(config.checkpoint_dir):
        print("|NOTE| Doesn't find the save dir, we will create a default one for you")
        os.mkdir(config.checkpoint_dir)

    if not os.path.isdir(config.output_data_dir):
        print("|NOTE| Doesn't find the output data dir, we will create a default one for you")
        os.mkdir(config.output_data_dir)

    return status

'''
set corresponding parameters of the corresponding model
'''
def show_args(config):
    print("=============== Command Line Tools Args ===============")
    for arg, value in vars(config).items():
        if isinstance(value, list):
            value = "[" + ",".join(list(map(str, value))) + "]"
        elif isinstance(value, bool):
            value = str(value)
        print("{:>20} <===> {:<20}".format(arg, value))
    print("=======================================================")


def preprocess():
    print("|LOGGING| Processing tokens and datasets...")
    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False)
    from itertools import islice
    columns = []

    with open(os.path.join(args.input_data_dir, args.train_filename)) as f_input:
        for row in islice(f_input, 0, 1):
            for _ in row.split("\t"):
                columns.append(_)

    x_feature = columns[0]
    y_feature = columns[1:]

    tv_datafields = [(x_feature, TEXT)]

    for col in y_feature:
        tv_datafields.append((col, LABEL))

    train_data, valid_data = TabularDataset.splits(path=args.input_data_dir,
                                                   format="tsv",
                                                   train=args.train_filename,
                                                   validation=args.valid_filename,
                                                   skip_header=True,
                                                   fields=tv_datafields)

    TEXT.build_vocab(train_data)

    pickle.dump(TEXT, open(os.path.join(args.output_data_dir, "TEXT.p"), "wb"))
    pickle.dump(y_feature, open(os.path.join(args.output_data_dir, "y_feature.p"), "wb"))
    print("FEATURE: ", len(y_feature))

    ############# pre-process done

    return train_data, valid_data, TEXT, x_feature, y_feature


def train(train_data, valid_data, TEXT, x_feature, y_feature, config):
    print("| Building batches...")
    device = torch.device("cuda:{}".format(config.master_device))
    # build dataloader
    train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                batch_size=config.batch_size * torch.cuda.device_count(),
                                device=device,
                                sort_key=lambda x: len(x.tokens),
                                sort_within_batch=True)

    train_dataloader = MiniBatchWrapper(train_iter, x_feature, y_feature)
    valid_dataloader = MiniBatchWrapper(valid_iter, x_feature, y_feature)

    print("| Building model...")

    if args.model == "LSTM":
        model = LSTM(config, vocab_size=len(TEXT.vocab),
                     label_num=len(y_feature),
                     device=device)
    elif args.model == "fastText":
        model = FastText(config, vocab_size=len(TEXT.vocab),
                         label_num=len(y_feature))
    elif args.model == "CNN":
        model = CNN(config, vocab_size=len(TEXT.vocab),
                    label_num=len(y_feature))

    model_args = model.get_args()

    if torch.cuda.device_count() > 1 and args.multi_gpu is True:
        print("Training on {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, optimizer, criterion)

    valid_loss_history = []

    print("| Training...")

    for epoch in range(1, args.epoch+1):
        trainer.train_step(train_dataloader, epoch, config)
        dev_loss = trainer.valid_step(model_args, valid_dataloader, epoch, config)
        if len(valid_loss_history) == 0 or dev_loss < valid_loss_history[0]:
            print("| Better checkpoint found !")
            torch.save({"model_type": args.model,
                        "model_args": model_args,
                        "model_state_dict": model.state_dict()},
                       os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
            valid_loss_history.append(dev_loss)
            valid_loss_history.sort()


if __name__ == "__main__":
    config = update_args(args)
    status = check_args(config)
    if status == True:
        show_args(config)
        tr, vl, TEXT, x_feature, y_feature = preprocess()
        train(tr, vl, TEXT, x_feature, y_feature, config)
