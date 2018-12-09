#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#tokenize = lambda x: x.split()
#TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
#LABEL = Field(sequential=False, use_vocab=False)
#
import pandas as pd

source = pd.read_csv("/data_hdd/zhihu/nlpcc2018/torchtext_data_full.tsv", sep="\t")
columns = list(source.columns)
#print(columns)
#
#x_feature = "tokens"
y_feature = []
#
#tv_datafields = [("tokens", TEXT)]
#
for col in columns[1:]:
#    tv_datafields.append((col, LABEL))
    y_feature.append(col)
#
#tst_datafields = [("tokens", TEXT)]
#
#source_data = TabularDataset(path="/data_hdd/zhihu/nlpcc2018/torchtext_data_full.tsv",
#                             format="tsv",
#                             skip_header=True,
#                             fields=tv_datafields)
#
#tst = TabularDataset(path="/data_hdd/zhihu/nlpcc2018/new_test.tsv", # the file path
#                     format='csv',
#                     skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
#                     fields=tst_datafields)
#
#train_data, valid_data = source_data.split()
#TEXT.build_vocab(source_data)
#
#class BatchWrapper:
#      def __init__(self, dl, x_var, y_vars):
#            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars
#
#      def __iter__(self):
#            for batch in self.dl:
#                # print(batch)
#                x = getattr(batch, self.x_var) # we assume only one input in this wrapper
#                if self.y_vars is  not None:
#                      temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
#                      y = torch.cat(temp, dim=1).float()
#                else:
#                      y = torch.zeros((1))
#                yield (x, y)
#
#      def __len__(self):
#            return len(self.dl)


class SimpleLSTM(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):
        super().__init__()
        self.embedding = nn.Embedding(483462, emb_dim)
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

import pickle
stoi = pickle.load(open("./text_stoi.p", "rb"))

model = SimpleLSTM(hidden_dim=100, emb_dim=300)
model.to("cuda")

model.load_state_dict(torch.load("./models/checkpoint_6.pt"))
model.eval()


sentence = "经济 房产 理财"
import numpy as np
def predict_sentiment(sentence):
    tokenized = sentence.split()
    indexed = [stoi[t] for t in tokenized]
    print(indexed)
    indexed = torch.LongTensor(indexed).to(device)
    indexed = indexed.unsqueeze(1)

    preds = model(indexed)
    preds = preds.data.cpu().numpy()
    preds = 1 / (1 + np.exp(-preds))
    preds = preds[0]
    index = np.argsort(preds)[::-1][:5]
    labels = [y_feature[idx] for idx in index]
    print(labels)


print(predict_sentiment(sentence))

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

