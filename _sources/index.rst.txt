.. caver documentation master file, created by
   sphinx-quickstart on Thu Jul  5 09:43:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to caver's doc!
=================================

`Caver` is a multilabel text classification toolkit based on `Pytorch`.

    Raising a torch to explore the cave.


Why we need this?
-------------------

Multilabel text classification is widely used in text-based recommendation
system, anti-spam, sentiment analysis and so on. Some basic machine learning
methods like SVM, Naive bayes can help us to find the category of text, but 
sentences usually belong to multi labels which is hard to extract all of them. 
Now we can use deep learning models to deal with it.

Features
---------

* Train text classifier
* Offer CNN, LSTM, HAN, SWEN etc.
* Get the feature vector of text
* Get the most possible labels for text
* Ensemble (soft voting)

Version
--------

* Python 3.5
* Pytorch 0.4


Tutorail
--------

Before train the model, you need to prepare text data in the format like:

.. code::

    __label__animal __label__sports The quick brown fox jumps over the lazy dog

Each line of text file cantains a list of labels and words separated by space.
Each label is start with '__label__'. 

If you use Chinese, please segment the sentence into words or single word first,
depends on which format you want.

.. code::

    __label__animal __label__sports 狐狸 跟 狗子 在 玩 跳山羊


Train
~~~~~~~~~~

::

    >>> from caver import Trainer
    >>> t = Trainer(
    >>>     'CNN',
    >>>     'data_path',
    >>>     ...... # kwargs will update the default value in config
    >>> )
    >>> t.train()

Classify
~~~~~~~~~~~

::

    >>> from caver import Caver
    >>> cnn = Caver('CNN', 'CNN_model.pth', 'data_path')

    # predict
    >>> cnn.predict('The quick brown fox jumps over the lazy dog')
    [1.0, 0.33, 0.15, 0.002, ......]

    # get top label
    >>> cnn.get_top_label('The quick brown fox jumps over the lazy dog')
    [['__label__animal', '__label__sports', ......],
    [1.0, 0.35, ......]]

    # ensemble
    >>> from caver import Ensemble
    >>> lstm = Caver('LSTM', 'LSTM_model.pth', 'data_path')
    >>> model = Ensemble([cnn, lstm])

    >>> model.predict('The quick brown fox jumps over the lazy dog', 'log')
    >>> model.get_top_label('The quick brown fox jumps over the lazy dog', 'avg')


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   detail
   model
   classify


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
