#!/usr/bin/env python
# encoding: utf-8

from caver import CaverModel

model_lstm = CaverModel("/data_hdd/caver_models/checkpoints_char_lstm", device="cpu")
model_cnn = CaverModel("/data_hdd/caver_models/checkpoints_char_cnn", device="cpu")

def predict(sentences):
    sent_char = []
    for sent in sentences:
        sent_char.append(" ".join(sent))

    labels_cnn = model_cnn.predict(sent_char, top_k=5)
    labels_lstm = model_lstm.predict(sent_char, top_k=5)
    return labels_cnn, labels_lstm

sentences = ["中美经济关系如何",
             "高等数学自学路线",
             "科比携手姚明出任2019篮球世界杯全球大使"]

labels = predict(sentences)

for _ in range(len(sentences)):
    print(sentences[_], labels[0][_], labels[1][_])


