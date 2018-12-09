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
from caver.utils import BatchWrapper
import arrow

parser = argparse.ArgumentParser(description="Caver training")
parser.add_argument("--model", type=str, choices=["CNN", "LSTM"],
                    help="choose the model", default="LSTM")
parser.add_argument("--input_data_dir", type=str, help="data dir")
parser.add_argument("--train_filename", type=str, default="train.tsv")
parser.add_argument("--valid_filename", type=str, default="valid.tsv")
parser.add_argument("--epoch", type=int, help="number of epoches", default=10)
parser.add_argument("--output_data_dir", type=str, default="processed_data")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="dir for checkpoints saving")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if os.path.isdir(args.checkpoint_dir) and len(os.listdir(args.checkpoint_dir)) != 0:
        status = False
        # exist but not empty
        print("save dir must be empty")

    if not os.path.isdir(args.checkpoint_dir):
        print("Doesn't find the save dir, we will create a default one for you")
        os.mkdir(args.checkpoint_dir)

    return status


def preprocess():
    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
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

#    tst_datafields = [("tokens", TEXT)]
#    # print(tv_datafields)
#    # print(y_feature)

    train_data, valid_data = TabularDataset.splits(path=args.input_data_dir,
                                                   format="tsv",
                                                   train=args.train_filename,
                                                   validation=args.valid_filename,
                                                   skip_header=True,
                                                   fields=tv_datafields)

#    tst = TabularDataset(path="/data_hdd/zhihu/nlpcc2018/new_test.tsv", # the file path
#                         format='csv',
#                         skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
#                         fields=tst_datafields)
##print("Unique tokens in TEXT vocabulary: {}".format(len(TEXT.vocab)))
## print("Unique tokens in LABEL vocabulary: {}".format(len(LABEL.vocab)))

#    # train_data, valid_data = source_data.split()
    TEXT.build_vocab(train_data)
    pickle.dump(TEXT, open(os.path.join(args.output_data_dir, "TEXT.p"), "wb"))
    pickle.dump(y_feature, open(os.path.join(args.output_data_dir, "y_feature.p"), "wb"))
#    TEXT = pickle.load(open("ttt.p", "rb"))
#    pickle.dump(train_data, open("train_ds.p", "wb"))
#    pickle.dump(valid_data, open("valid_ds.p", "wb"))
#
#    train_data = pickle.load(open("train_ds.p", "rb"))
#    valid_data = pickle.load(open("valid_ds.p", "rb"))

    train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                batch_size=64,
                                device="cuda",
                                sort_key=lambda x: len(x.tokens),
                                sort_within_batch=True)
    print(arrow.now())

    train_dataloader = BatchWrapper(train_iter, x_feature, y_feature)
    valid_dataloader = BatchWrapper(valid_iter, x_feature, y_feature)


    return train_dataloader, valid_dataloader, TEXT


def train(train_data, valid_data, TEXT):
    model = LSTM(hidden_dim=300, embedding_dim=256, vocab_size=len(TEXT.vocab), label_num=400, device="cuda", layer_num=1)
    model.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    valid_loss_history = []

    for epoch in range(1, args.epoch+1):
        train_step(model, train_data, optimizer, criterion, epoch)
        valid_step(model, valid_data, criterion, valid_loss_history, epoch)


def train_step(model, train_data, opt, criterion, epoch):
    running_loss = 0.0
    num_sample = 0
    model.train()
    tqdm_progress = tqdm.tqdm(train_data, desc="| Training epoch {}/{}".format(epoch, args.epoch))
    for x, y in tqdm_progress:
        opt.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()

        num_sample += x.size(0)
        running_loss += loss.item()*x.size(0)
        # ave_loss = running_loss / len(train_data)

        tqdm_progress.set_postfix({"Ave Loss ":"{:.4f}".format(running_loss / num_sample)})
    # calculate the validation loss for this epoch


def valid_step(model, valid_data, criterion, valid_loss_history, epoch):
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
        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, "checkpoint_{}.pt".format(epoch)))
        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
        valid_loss_history.append(loss.item())
        valid_loss_history.sort()

#
## kk = LABEL.vocab.stoi
## print(kk)


#test_iter = Iterator(tst, batch_size=1, device="cuda", sort=False, sort_within_batch=False, repeat=False, train=False)
#
#
#test_dl = BatchWrapper(test_iter, x_feature, None)
#
#sample = next(train_dl.__iter__())
#
## print(sample[0].shape)
## print(sample[1].shape)
#
#
#class SimpleLSTM(nn.Module):
#    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
#        super().__init__()
#        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
#        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
#        self.linear_layers= []
#
#        for _ in range(num_linear - 1):
#            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
#            self.linear_layers = nn.ModuleList(self.linear_layers)
#
#        self.predictor = nn.Linear(hidden_dim, 400)
#
#    def forward(self, seq):
#        hdn, _ = self.encoder(self.embedding(seq))
#        feature = hdn[-1, :, :]
#        for layer in self.linear_layers:
#            feature = layer(feature)
#        preds = self.predictor(feature)
#        return preds
#
#
## model =
#
#
#import numpy as np
#def test_step():
#    model.eval()
#    for x, y in tqdm.tqdm(test_dl):
#        preds = model(x)
#        preds = preds.data.cpu().numpy()
#        preds = 1 / (1 + np.exp(-preds))
#        preds = preds[0]
#        index = np.argsort(preds)[::-1][:5]
#        # index = index[0]
#        labels = [y_feature[idx] for idx in index]
#        print(labels)
#        # test_preds.append(preds)
#        # test_preds = np.hstack(test_preds)
#
#
#N_EPOCHS = 50
#for epoch in range(N_EPOCHS):
#    print("doing----", epoch)
#    train_step()
#    valid_step(epoch)
#    test_step()

if __name__ == "__main__":
    status = check_args()
    if status == True:
        tr, vl, TEXT = preprocess()
        train(tr, vl, TEXT)
