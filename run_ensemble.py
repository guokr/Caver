# -*- coding: UTF-8 -*-
from __future__ import print_function
import sys
import caver.utils as utils
import argparse
from caver import CaverModel
from caver import EnsembleModel
"""
note:
1.if one model dir arg is empty, the model is not ensemble
2.three soft voting method,include mean，hmean，gmean. And if model_ratio arg is not empty, use weighted voting.
3.now each chinese character is split by space, because is used for char models, not for word model

"""
parser = argparse.ArgumentParser(description="Caver Ensemble")
parser.add_argument(
    "--cnn",
    type=str,
    help="CNN model dir",
    default="/data_hdd/caver_models/checkpoints_char_cnn",
)
parser.add_argument(
    "--lstm",
    type=str,
    help="LSTM model dir",
    default="/data_hdd/caver_models/checkpoints_char_lstm",
)
parser.add_argument("--fasttext", type=str, help="fastText model dir", default="")
parser.add_argument(
    "--voting",
    type=str,
    choices=["mean", "hmean", "gmean"],
    help="voting choice",
    default="mean",
)
parser.add_argument(
    "--model_ratio",
    type=dict,
    help="model ratio, sum equal one",
    default={"cnn": 0.5, "lstm": 0.5},
    # default={},
)
parser.add_argument(
    "--sentences",
    type=list,
    help="sentences list include sentence which each chinese character split by space",
    default=[
        "中 美 经 济 关 系 如 何",
        "看 美 剧 学 英 语 靠 谱 吗",
        "科 比 携 手 姚 明 出 任 2019 篮 球 世 界 杯 全 球 大 使",
        "如 何 在 《 权 力 的 游 戏 》中 苟 到 最 后",
        "英 雄 联 盟 LPL 夏 季 赛 RNG 能 否 击 败 TOP 战 队",
    ],
)
parser.add_argument(
    "--top_k", type=int, help="get top k labels from predict", default=5
)

args = parser.parse_args()


if __name__ == "__main__":
    status = utils.check_ensemble_args(args)
    if status:
        utils.show_ensemble_args(args)
        models = []
        model_ratio = []
        if args.lstm:
            model_lstm = CaverModel(args.lstm, device="cpu")
            models.append(model_lstm)
            if len(args.model_ratio) > 0:
                model_ratio.append(args.model_ratio["lstm"])
        if args.cnn:
            model_cnn = CaverModel(args.cnn, device="cpu")
            models.append(model_cnn)
            if len(args.model_ratio) > 0:
                model_ratio.append(args.model_ratio["cnn"])

        # later add other models, such as fasttext

        model_ensemble = EnsembleModel(models, model_ratio)

        labels = model_ensemble.predict(
            args.sentences, top_k=args.top_k, method=args.voting
        )
        for _ in range(len(args.sentences)):
            print(args.sentences[_], labels[_])
