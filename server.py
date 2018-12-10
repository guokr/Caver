#!/usr/bin/env python
# encoding: utf-8

import torch

from caver.model import LSTM, FastText
import dill as pickle
import argparse
import os
import sys
parser = argparse.ArgumentParser(description="Caver Serving")
parser.add_argument("--output_data_dir", type=str, help="data dir")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--model_file", type=str, default="checkpoint_best.pt")
parser.add_argument("--number_labels", type=int, default=5,
                    help="number of labels we need to predict")

args = parser.parse_args()

y_feature = pickle.load(open(os.path.join(args.output_data_dir, "y_feature.p"), "rb"))
TEXT = pickle.load(open(os.path.join(args.output_data_dir, "TEXT.p"), "rb"))

loaded_checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.model_file))

model_type = loaded_checkpoint["model_type"]

if model_type == "LSTM":
    model = LSTM()
elif model_type == "fastText":
    model = FastText()
elif model_type == "CNN":
    model = FastText()
else:
    sys.exit()

model.update_args(loaded_checkpoint["model_args"])
model.to(args.device)
model.load_state_dict(loaded_checkpoint["model_state_dict"])
model.eval()

import numpy as np
import arrow

def single_predict(sentence):
    start = arrow.now()
    tokenized = sentence.split()
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    indexed = torch.LongTensor(indexed).to(args.device)
    indexed = indexed.unsqueeze(1)
    preds = model(indexed)
    preds = preds.data.cpu().numpy()
    preds = 1 / (1 + np.exp(-preds))
    preds = preds[0]
    index = np.argsort(preds)[::-1][:args.number_labels]
    labels = [y_feature[idx] for idx in index]
    end = arrow.now()
    print("sentence: {} ==> predicted labels: {} used {:.4f}seconds".format(sentence, ",".join(labels), (end-start).total_seconds()))

sentences = ["经济 理财",
             "数学 高等数学",
             "篮球 篮球场"]

for sent in sentences:
    single_predict(sent)
