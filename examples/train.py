#!/usr/bin/env python
# encoding: utf-8

# import argparse
import torch
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

# parser = argparse.ArgumentParser(description="Caver training")
# parser.add_argument("--model", type=str, choices=["CNN", "LSTM"],
#                     help="choose the model", default="CNN")
# args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import pickle
# word2index = pickle.load(open("dest_dir/word2index.p", "rb"))
# index2word = pickle.load(open("dest_dir/index2word.p", "rb"))
# tag2index = pickle.load(open("dest_dir/tag2index.p", "rb"))
# index2tag = pickle.load(open("dest_dir/index2tag.p", "rb"))

tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)


import pandas as pd

source = pd.read_csv("/data_hdd/zhihu/nlpcc2018/torchtext_data_full.tsv", sep="\t")
columns = list(source.columns)
print(columns)

x_feature = "tokens"
y_feature = []

tv_datafields = [("tokens", TEXT)]

for col in columns[1:]:
    tv_datafields.append((col, LABEL))
    y_feature.append(col)

tst_datafields = [("tokens", TEXT)]
# print(tv_datafields)
# print(y_feature)

source_data = TabularDataset(path="/data_hdd/zhihu/nlpcc2018/torchtext_data_full.tsv",
                             format="tsv",
                             skip_header=True,
                             fields=tv_datafields)

tst = TabularDataset(path="/data_hdd/zhihu/nlpcc2018/new_test.tsv", # the file path
                     format='csv',
                     skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                     fields=tst_datafields)

train_data, valid_data = source_data.split()
TEXT.build_vocab(source_data)
# LABEL.build_vocab(source_data)

print("Unique tokens in TEXT vocabulary: {}".format(len(TEXT.vocab)))
# print("Unique tokens in LABEL vocabulary: {}".format(len(LABEL.vocab)))
# kk = LABEL.vocab.stoi
# print(kk)

train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                               # split_ratio=0.85,
                            batch_size=128,
                            device="cuda",
                            sort_key=lambda x: len(x.tokens),
                            sort_within_batch=True,)

test_iter = Iterator(tst, batch_size=1, device="cuda", sort=False, sort_within_batch=False, repeat=False, train=False)

class BatchWrapper:
      def __init__(self, dl, x_var, y_vars):
            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

      def __iter__(self):
            for batch in self.dl:
                # print(batch)
                x = getattr(batch, self.x_var) # we assume only one input in this wrapper
                if self.y_vars is  not None:
                      temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                      y = torch.cat(temp, dim=1).float()
                else:
                      y = torch.zeros((1))
                yield (x, y)

      def __len__(self):
            return len(self.dl)

train_dl = BatchWrapper(train_iter, x_feature, y_feature)
valid_dl = BatchWrapper(valid_iter, x_feature, y_feature)
test_dl = BatchWrapper(test_iter, x_feature, None)

sample = next(train_dl.__iter__())

# print(sample[0].shape)
# print(sample[1].shape)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.linear_layers= []

        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)

        self.predictor = nn.Linear(hidden_dim, 400)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds


model = SimpleLSTM(hidden_dim=100, emb_dim=300)
model.to("cuda")

import tqdm

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.BCEWithLogitsLoss()


def train_step():
    running_loss = 0.0
    model.train()
    for x, y in tqdm.tqdm(train_dl):
        opt.zero_grad()
        preds = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()
        running_loss += loss.item()*x.size(0)
    # calculate the validation loss for this epoch
    print("training loss:", running_loss / len(train_iter))


def valid_step(epoch):
    running_loss = 0.0
    model.eval()
    for x, y in tqdm.tqdm(valid_dl):
        preds = model(x)
        loss = loss_func(preds, y)
        running_loss += loss.item()*x.size(0)
    print("validation loss:", running_loss / len(valid_iter))
    torch.save(model.state_dict(), "models/checkpoint_{}.pt".format(epoch))


import numpy as np
def test_step():
    model.eval()
    for x, y in tqdm.tqdm(test_dl):
        preds = model(x)
        preds = preds.data.cpu().numpy()
        preds = 1 / (1 + np.exp(-preds))
        preds = preds[0]
        index = np.argsort(preds)[::-1][:5]
        # index = index[0]
        labels = [y_feature[idx] for idx in index]
        print(labels)
        # test_preds.append(preds)
        # test_preds = np.hstack(test_preds)


N_EPOCHS = 50
for epoch in range(N_EPOCHS):
    print("doing----", epoch)
    train_step()
    valid_step(epoch)
    test_step()
