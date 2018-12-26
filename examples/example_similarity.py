#!/usr/bin/env python
# encoding: utf-8

from caver import CaverModel
from sklearn.metrics.pairwise import cosine_similarity

model_lstm = CaverModel("/data_hdd/caver_models/checkpoints_char_lstm", device="cpu")

def get_vectors(sent_char):
    vectors = model_lstm.predict_prob(sent_char)
    return vectors

sentences = ["西 部 世 界 第 二 季 第 十 集 说 文 解 图",
             "你 在 守 望 先 锋  中 最 秀 的 一 波 操 作 是 什 么 样 子",
             "如 何 在 《 权 力 的 游 戏 》中 苟 到 最 后",
             "英 雄 联 盟 LPL 夏 季 赛 RNG 能 否 击 败 TOP 战 队"]

vectors = get_vectors(sentences)

similarities = cosine_similarity(vectors)

print(similarities)
