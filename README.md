# Caver: a toolkit for multilabel text classification.

[中文版](./README_zh.md)

---

Rising a torch in the cave to see the words on the wall. This is the `Caver`.

[Documents](https://guokr.github.io/Caver)

## Requirements

* PyTorch
* tqdm
* torchtext
* scipy
* numpy
* Python3

## How to train on your own dataset

```
$python3 train.py --input_data_dir {path to your origin dataset}
                  --output_data_dir {path to store the preprocessed dataset}
                  --train_filename train.tsv
                  --valid_filename valid.tsv
                  --checkpoint_dir {path to save the checkpoints}
                  --model {fastText/CNN/LSTM}
                  --batch_size {16, you can modify this for you own}
                  --epoch {10}

```

## Did you guys have some pre-trained models

Yes, the download link will be available in soon

## How to setup the models for inference

Basicly just setup the model and target labels, you can check examples in server.py and ensemble.py
