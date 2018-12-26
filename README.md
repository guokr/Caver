<h1 align="center">Caver</h1>

<p align="center">Rising a torch in the cave to see the words on the wall, tag your short text in 3 lines. Caver uses Facebook's <a href="https://pytorch.org/">PyTorch</a> project to make the implementation easier.</p>

<p align="center">
  <a href="https://pypi.org/project/caver/">
      <img src="https://img.shields.io/pypi/v/caver.svg?colorB=brightgreen"
           alt="Pypi package">
    </a>
  <a href="https://github.com/guokr/caver/releases">
      <img src="https://img.shields.io/github/release/guokr/caver.svg"
           alt="GitHub release">
  </a>
  <a href="https://github.com/guokr/caver/issues">
        <img src="https://img.shields.io/github/issues/guokr/caver.svg"
             alt="GitHub issues">
  </a>
  <a href="https://travis-ci.org/guokr/Caver/">
    <img src="https://travis-ci.org/guokr/Caver.svg"
         alt="Travis CI">
  </a>
</p>

<p align="center">
  <a href="#quick-demo">Demo</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#install">Install</a> •
  <a href="#did-you-guys-have-some-pre-trained-models">Pre-trained models</a> •
  <a href="#how-to-train-on-your-own-dataset">Train</a> •
  <a href="#more-examples">Examples</a> •
  <a href="https://guokr.github.io/Caver/">Document</a>
</p>

<p align="center">
  <img src=".github/demo.gif?raw=true" width="550">
 </p>

<h2 align="center">Quick Demo</h2>

```python
from caver import CaverModel
model = CaverModel("./checkpoint_path")

sentence = ["看 美 剧 学 英 语 靠 谱 吗",
            "科 比 携 手 姚 明 出 任 2019 篮 球 世 界 杯 全 球 大 使",
            "如 何 在 《 权 力 的 游 戏 》 中 苟 到 最 后",
            "英 雄 联 盟 LPL 夏 季 赛 RNG 能 否 击 败 TOP 战 队"]

model.predict(sentence[0], top_k=3)
>>> ['美剧', '英语', '英语学习']

model.predict(sentence[1], top_k=5)
>>> ['篮球', 'NBA', '体育', 'NBA 球员', '运动']

model.predict(sentence[2]. top_k=7)
>>> ['权力的游戏（美剧）', '美剧', '影视评论', '电视剧', '电影', '文学', '小说']

model.predict(sentence[3], top_k=6)
>>> ['英雄联盟（LoL）', '电子竞技', '英雄联盟职业联赛（LPL）', '游戏', '网络游戏', '多人联机在线竞技游戏 (MOBA)']
```

<h2 align="center">Requirements</h2>

* PyTorch
* tqdm
* torchtext
* numpy
* Python3

<h2 align="center">Install</h2>

```bash
$ pip install caver --user
```

<h2 align="center">Did you guys have some pre-trained models</h2>

Yes, we have released two pre-trained models on Zhihu NLPCC2018 [opendataset](http://tcci.ccf.org.cn/conference/2018/taskdata.php).

If you want to use the pre-trained model for performing text tagging, you can download it (along with other important inference material) from the Caver releases page. Alternatively, you can run the following command to download and unzip the files in your current directory:

```bash
$ wget -O - https://github.com/guokr/Caver/releases/download/0.1/checkpoints_char_cnn.tar.gz | tar zxvf -
$ wget -O - https://github.com/guokr/Caver/releases/download/0.1/checkpoints_char_lstm.tar.gz | tar zxvf -
```

<h2 align="center">How to train on your own dataset</h2>

```bash
$ python3 train.py --input_data_dir {path to your origin dataset}
                   --output_data_dir {path to store the preprocessed dataset}
                   --train_filename train.tsv
                   --valid_filename valid.tsv
                   --checkpoint_dir {path to save the checkpoints}
                   --model {fastText/CNN/LSTM}
                   --batch_size {16, you can modify this for you own}
                   --epoch {10}

```

<h2 align="center">More Examples</h2>

It's updating, but basically you can check [examples](https://github.com/guokr/Caver/tree/master/examples).
