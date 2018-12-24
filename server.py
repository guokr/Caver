#!/usr/bin/env python
# encoding: utf-8

import torch
from caver.model import CNN, LSTM

cnn_checkpoint_dir = "/data_hdd/caver_models/checkpoints_char_cnn"
lstm_checkpoint_dir = "/data_hdd/caver_models/checkpoints_char_lstm"

device = torch.device("cpu")

model_lstm = LSTM()
model_lstm.load(lstm_checkpoint_dir)
model_lstm.to(device)
model_lstm.eval()

model_cnn = CNN()
model_cnn.load(cnn_checkpoint_dir)
model_cnn.to(device)
model_cnn.eval()

sentences_char = ["经 济",
                  "数 学 高 等 数 学",
                  "篮 球 篮 球 场"]

def predict(sentences):
    labels_cnn = model_cnn.predict(sentences,top_k=5)
    labels_lstm = model_lstm.predict(sentences,top_k=5)
    return labels_cnn, labels_lstm

labels = predict(sentences_char)
print(labels)
