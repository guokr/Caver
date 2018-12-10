#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import dill as pickle

from torchtext.data import Field, TabularDataset, BucketIterator
from caver.model import LSTM
from caver.utils import MiniBatchWrapper

parser = argparse.ArgumentParser(description="Caver training")
parser.add_argument("--model", type=str, choices=["CNN", "LSTM"],
                    help="choose the model", default="LSTM")
parser.add_argument("--input_data_dir", type=str, help="data dir")
parser.add_argument("--train_filename", type=str, default="train.tsv")
parser.add_argument("--valid_filename", type=str, default="valid.tsv")
parser.add_argument("--epoch", type=int, help="number of epoches", default=10)
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size for each GPU card")
parser.add_argument("--output_data_dir", type=str, default="processed_data")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="dir for checkpoints saving")
parser.add_argument("--model_meta", type=str, default="model.pkl",
                    help="filename to store the model class instance")
parser.add_argument("--master_device", type=int, default=0)
parser.add_argument("--multi_gpu", action="store_true")

args = parser.parse_args()

def check_args():
    status = True
    if not os.path.exists(os.path.join(args.input_data_dir, args.train_filename)):
        status = False
        print("train file doesn't exist")

    if not os.path.exists(os.path.join(args.input_data_dir, args.valid_filename)):
        status = False
        print("valid file doesn't exist")

    if torch.cuda.is_available() == False:
        status = False
        print("Currently we dont support CPU training")

    if torch.cuda.device_count() == 1 and args.multi_gpu == True:
        status = False
        print("We only detected {} GPU".format(torch.cuda.device_count()))

    if os.path.isdir(args.checkpoint_dir) and len(os.listdir(args.checkpoint_dir)) != 0:
        status = False
        # exist but not empty
        print("save dir must be empty")

    if not os.path.isdir(args.checkpoint_dir):
        print("Doesn't find the save dir, we will create a default one for you")
        os.mkdir(args.checkpoint_dir)

    print(args)

    return status


def preprocess():
    print("| Processing tokens and datasets...")
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

    ############# pre-process done

    return train_data, valid_data, TEXT, x_feature, y_feature

def train(train_data, valid_data, TEXT, x_feature, y_feature):

    print("| Building batches...")

    device = torch.device("cuda:{}".format(args.master_device))

    # build dataloader
    train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                batch_size=args.batch_size * torch.cuda.device_count(),
                                device=device,
                                sort_key=lambda x: len(x.tokens),
                                sort_within_batch=True)

    train_dataloader = MiniBatchWrapper(train_iter, x_feature, y_feature)
    valid_dataloader = MiniBatchWrapper(valid_iter, x_feature, y_feature)

    print("| Building model...")

    model = LSTM(hidden_dim=300, embedding_dim=256,
                 vocab_size=len(TEXT.vocab),
                 label_num=len(y_feature),
                 device=device, layer_num=2)
    model_args = model.get_args()


    if torch.cuda.device_count() > 1 and args.multi_gpu is True:
        print("Training on {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    valid_loss_history = []

    print("| Training...")

    for epoch in range(1, args.epoch+1):
        train_step(model, train_dataloader, optimizer, criterion, epoch)
        valid_step(model, model_args, valid_dataloader, criterion, valid_loss_history, epoch)


def train_step(model, train_data, opt, criterion, epoch):
    running_loss = 0.0
    num_sample = 0
    model.train()
    tqdm_progress = tqdm.tqdm(train_data, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    for x, y in tqdm_progress:
        opt.zero_grad()
        # print(x.shape)
        # print(y.shape)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()

        num_sample += x.size(0)
        running_loss += loss.item()*x.size(0)
        # ave_loss = running_loss / len(train_data)

        tqdm_progress.set_postfix({"Ave Loss ":"{:.4f}".format(running_loss / num_sample)})
    # calculate the validation loss for this epoch


def valid_step(model, model_args, valid_data, criterion, valid_loss_history, epoch):
    running_loss = 0.0
    num_sample = 0
    model.eval()
    tqdm_progress = tqdm.tqdm(valid_data, desc="| Validating epoch {}/{}".format(epoch, args.epoch))
    for x, y in tqdm_progress:
        preds = model(x)
        loss = criterion(preds, y)

        num_sample += x.size(0)
        running_loss += loss.item()*x.size(0)

        # ave_loss = running_loss / len(valid_data)
        tqdm_progress.set_postfix({"Ave Loss ":"{:.4f}".format(running_loss / num_sample)})

    if len(valid_loss_history) == 0 or loss.item() < valid_loss_history[0]:
        torch.save({"model_args": model_args, "model_state_dict": model.state_dict()},
                   os.path.join(args.checkpoint_dir, "checkpoint_{}.pt".format(epoch)))

        torch.save({"model_args": model_args, "model_state_dict": model.state_dict()},
                   os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
        valid_loss_history.append(loss.item())
        valid_loss_history.sort()


if __name__ == "__main__":
    status = check_args()
    if status == True:
        tr, vl, TEXT, x_feature, y_feature = preprocess()
        train(tr, vl, TEXT, x_feature, y_feature)
