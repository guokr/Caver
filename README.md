# Caver: a toolkit for multilabel text classification.

Rising a torch in the cave to see the words on the wall. This is the `Caver`.

## Tutorial

[Documents](https://guokr.github.io/Caver)

### Train model

```python
from caver import Trainer
t = Trainer(
    'CNN',
    'data_path',
    ...... # kwargs will update the default value in config
)
t.train()
```

### Classify

```python
from caver import Caver
cnn = Caver('CNN', 'CNN_model.pth', 'data_path')

# predict
cnn.predict('Across the Great Wall, we can reach every corner in the world')

# get top label
cnn.get_top_label('The quick brown fox jumps over the lazy dog')

# ensemble
from caver import Ensemble
swen = Caver('SWEN', 'SWEN_model.pth', 'data_path')
model = Ensemble([cnn, swen])

model.predict('The quick brown fox jumps over the lazy dog', 'log')
model.get_top_label('The quick brown fox jumps over the lazy dog', 'avg')
```


## TODO

* [x] BaseModule
* [x] Data
* [x] classify
* [x] ensemble: voting
* [x] config
* [x] model save and load
* [x] models: CNN, LSTM, SWEN, HAN
* [x] dropout
* [ ] fastText support
* [ ] docker
