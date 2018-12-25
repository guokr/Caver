<h1 align="center">Caver<h1>

<p align="center">Rising a torch in the cave to see the words on the wall. Tag short text in 3 lines. This is the **Caver**.</p>

<p align="center">
  <a href="https://pypi.org/search/?q=bert-serving">
      <img src="https://img.shields.io/pypi/v/caver.svg?colorB=brightgreen"
           alt="Pypi package">
    </a>
  <a href="https://github.com/hanxiao/bert-as-service/releases">
      <img src="https://img.shields.io/github/release/guokr/caver.svg"
           alt="GitHub release">
  </a>
  <a href="https://github.com/hanxiao/bert-as-service/issues">
        <img src="https://img.shields.io/github/issues/guokr/caver.svg"
             alt="GitHub issues">
  </a>
</p>

<p align="center">
  <a href="#Quick Demo">Quick Demo</a> •
  <a href="#what-is-it">Requirements</a> •
  <a href="#install">Install</a> •
  <a href="#book-tutorial">Pre-trained models</a> •
  <a href="#speech_balloon-faq">Train</a> •
  <a href="#zap-benchmark">Examples</a>
  
</p>
<p align="center">
  <img src=".github/demo.gif?raw=true" width="700">
 </p>


<h2 align="center">Quick Demo</h2>

```python
from caver import CaverModel
model = CaverModel("./checkpoint_path", device="cpu")

sentence = ["看 英 语 学 美 剧 靠 谱 吗", "科 比 携 手 姚 明 出 任 2019 篮 球 世 界 杯 全 球 大 使"]

model.predict(sentence[0], top_k=3)
>>> ['英语学习', '英语', '美剧']

model.predict(sentence[1], top_k=10)
>>> ['篮球', 'NBA', '体育', 'NBA 球员', '运动']
```

[Documents](https://guokr.github.io/Caver)

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

Yes, we have released two pre-trained models on Zhihu NLPCC2018 open dataset.

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

<h2 align="center">How to setup the models for inference</h2>
Basically just setup the model and target labels, you can check [examples](./examples).
