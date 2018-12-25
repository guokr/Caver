#!/usr/bin/env python
# encoding: utf-8

from caver import CaverModel, EnsembleModel

model_lstm = CaverModel("/data_hdd/caver_models/checkpoints_char_lstm", device="cpu")
model_cnn = CaverModel("/data_hdd/caver_models/checkpoints_char_cnn", device="cpu")

lstm_cnn_log = EnsembleModel([model_lstm, model_cnn])

def predict(sentences):
    sent_char = []
    for sent in sentences:
        sent_char.append(" ".join(sent))

    labels = lstm_cnn_log.predict(sent_char,
                                  top_k=5,
                                  method="gmean")
    return labels

sentences_char = ["中美经济关系如何",
                  "高等数学自学路线",
                  "科比携手姚明出任2019篮球世界杯全球大使"]

labels = predict(sentences_char)

for _ in range(len(sentences_char)):
    print(sentences_char[_], labels[_])

