#!/usr/bin/env python
# encoding: utf-8

import torch

from caver.model import LSTM
import dill as pickle
import argparse
import os

parser = argparse.ArgumentParser(description="Caver Serving")
parser.add_argument("--output_data_dir", type=str, help="data dir")
parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()

y_feature = pickle.load(open(os.path.join(args.output_data_dir, "y_feature.p"), "rb"))
TEXT = pickle.load(open(os.path.join(args.output_data_dir, "TEXT.p"), "rb"))

model = LSTM(hidden_dim=300, embedding_dim=256, vocab_size=len(TEXT.vocab), label_num=400, device="cuda", layer_num=1)
model.to(args.device)

model.load_state_dict(torch.load("./checkpoints/checkpoint_9.pt"))
model.eval()

import numpy as np
import arrow
def predict_sentiment(sentence):
    start = arrow.now()
    tokenized = sentence.split()
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    indexed = torch.LongTensor(indexed).to(args.device)
    indexed = indexed.unsqueeze(1)
    preds = model(indexed)
    preds = preds.data.cpu().numpy()
    preds = 1 / (1 + np.exp(-preds))
    preds = preds[0]
    index = np.argsort(preds)[::-1][:5]
    labels = [y_feature[idx] for idx in index]
    end = arrow.now()
    print("sentence: {} ==> predicted labels: {} used {:.4f}seconds".format(sentence, ",".join(labels), (end-start).total_seconds()))

sentence = "经济 房产 理财"
predict_sentiment(sentence)
