#!/usr/bin/env python
# encoding: utf-8

import torch
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

import pandas as pd

source = pd.read_csv("/data_hdd/zhihu/nlpcc2018/nn.tsv", sep="\t")
columns = list(source.columns)
print(columns)

x_feature = "tokens"
y_feature = []

tv_datafields = [("tokens", TEXT)]

for col in columns[1:]:
    tv_datafields.append((col, LABEL))
    y_feature.append(col)

source_data = TabularDataset(path="/data_hdd/zhihu/nlpcc2018/nn.tsv",
                             format="tsv",
                             skip_header=True,
                             fields=tv_datafields)
TEXT.build_vocab(source_data)
# LABEL.build_vocab(source_data)
print(len(source_data))
print(vars(source_data.examples[0]))

print("Unique tokens in TEXT vocabulary: {}".format(len(TEXT.vocab)))
# print("Unique tokens in LABEL vocabulary: {}".format(len(LABEL.vocab)))

#train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
#                            batch_size=1,
#                            device="cuda",
#                            sort_key=lambda x: len(x.tokens),
#                            sort_within_batch=True,)

train_iter = Iterator(source_data, batch_size=1)

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

count = 0
for x, y in train_dl:
    count += 1
    print(x, y)
    if count == 2:
        break
