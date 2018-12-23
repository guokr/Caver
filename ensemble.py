#!/usr/bin/env python
# encoding: utf-8

import torch

from caver.model import LSTM, FastText, CNN
import dill as pickle
import argparse
import os
import sys

#parser = argparse.ArgumentParser(description="Caver Serving")
#parser.add_argument("--output_data_dir", type=str, help="data dir")
#parser.add_argument("--device", type=str, default="cpu")
## parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
#parser.add_argument("--model_file", type=str, default="checkpoint_best.pt")
#parser.add_argument("--number_labels", type=int, default=5, help="number of labels we need to predict")
#parser.add_argument("--token_type", type=str, choices=["char", "word"],
#                    help="token type for sample prediction")
#
#args = parser.parse_args()
output_data_dir ="/data_hdd/caver_models/dataset_meta_char"

y_feature = pickle.load(open(os.path.join(output_data_dir, "y_feature.p"), "rb"))
TEXT = pickle.load(open(os.path.join(output_data_dir, "TEXT.p"), "rb"))
device = torch.device("cpu")

# loaded_checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.model_file), map_location=device)
# model_type = loaded_checkpoint["model_type"]

model_lstm = LSTM()
model_cnn = CNN()

loaded_checkpoint_lstm = torch.load("/data_hdd/caver_models/checkpoints_char_lstm/checkpoint_best.pt",
                                    map_location=device)
loaded_checkpoint_cnn = torch.load("/data_hdd/caver_models/checkpoints_char_cnn/checkpoint_best.pt",
                                   map_location=device)

#if model_type == "LSTM":
#    model = LSTM()
#elif model_type == "fastText":
#    model = FastText()
#elif model_type == "CNN":
#    model = CNN()
#else:
#    sys.exit()

model_lstm.update_args(loaded_checkpoint_lstm["model_args"])
model_lstm.load_state_dict(loaded_checkpoint_lstm["model_state_dict"])
model_cnn.update_args(loaded_checkpoint_cnn["model_args"])
model_cnn.load_state_dict(loaded_checkpoint_cnn["model_state_dict"])

model_lstm.to(device)
model_lstm.eval()

model_cnn.to(device)
model_cnn.eval()

from caver import Ensemble

lstm_cnn_log = Ensemble([model_lstm, model_cnn])
# print(lstm_cnn_log)

def predict(sentences):
    labels = []
    preds = lstm_cnn_log.predict(sentences, TEXT.vocab.stoi, top_k=5)
    preds = preds.data.cpu().numpy()
    for pred in preds:
        labels.append([y_feature[idx] for idx in pred])
    res = list(zip(sentences, labels))

    for ele in res:
        print("sentence: {} ==> predicted labels: {}".format(ele[0], ",".join(ele[1])))


sentences_char = ["经 济",
                  "数 学 高 等 数 学",
                  "篮 球 篮 球 场"]
#
#sentences_word = ["经济",
#                  "数学 高等数学",
#                  "篮球 篮球场"]
#
predict(sentences_char)
