#!/usr/bin/env python
# encoding: utf-8

from caver import CaverModel

model_lstm = CaverModel("/data_hdd/caver_models/checkpoints_char_lstm", device="cpu")
model_cnn = CaverModel("/data_hdd/caver_models/checkpoints_char_cnn", device="cpu")

def predict(sent_char):
    labels_cnn = model_cnn.predict(sent_char, top_k=5)
    labels_lstm = model_lstm.predict(sent_char, top_k=10)
    return labels_cnn, labels_lstm

sentences = ["中 美 经 济 关 系 如 何",
             "看 美 剧 学 英 语 靠 谱 吗",
             "科 比 携 手 姚 明 出 任 2019 篮 球 世 界 杯 全 球 大 使",
             "如 何 在 《 权 力 的 游 戏 》中 苟 到 最 后",
             "英 雄 联 盟 LPL 夏 季 赛 RNG 能 否 击 败 TOP 战 队"]

labels = predict(sentences)

for _ in range(len(sentences)):
    print(sentences[_], labels[0][_], labels[1][_])


