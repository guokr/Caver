#!/usr/bin/env python
# encoding: utf-8

import torch

from caver.model import LSTM, CNN

cnn_checkpoint_dir = "/data_hdd/caver_models/checkpoints_char_cnn"
lstm_checkpoint_dir = "/data_hdd/caver_models/checkpoints_char_lstm"

device = torch.device("cpu")

model_cnn = CNN()
model_cnn.load(cnn_checkpoint_dir)
model_lstm = LSTM()
model_lstm.load(lstm_checkpoint_dir)

model_cnn.to(device)
model_cnn.eval()
model_lstm.to(device)
model_lstm.eval()

from caver import Ensemble

lstm_cnn_log = Ensemble([model_lstm, model_cnn])

def predict(sentences):
    labels = lstm_cnn_log.predict(sentences,
                                  top_k=5,
                                  method="gmean")
    return labels

sentences_char = ["经 济",
                  "数 学 高 等 数 学",
                  "篮 球 篮 球 场"]

labels = predict(sentences_char)
print(labels)
