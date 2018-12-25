#!/usr/bin/env python
# encoding: utf-8

from caver import CaverModel

model_lstm = CaverModel("/data_hdd/caver_models/checkpoints_char_lstm", device="cpu")
model_cnn = CaverModel("/data_hdd/caver_models/checkpoints_char_cnn", device="cpu")

def predict(sentences):
    labels_cnn = model_cnn.predict(sentences, top_k=5)
    labels_lstm = model_lstm.predict(sentences, top_k=10)
    return labels_cnn, labels_lstm

sentences = ["看 英 语 学 美 剧 靠 谱 吗",
             "科 比 携 手 姚 明 出 任 2019 篮 球 世 界 杯 全 球 大 使"]

labels = predict(sentences)

for _ in range(len(sentences)):
    print(sentences[_], labels[0][_], labels[1][_])


