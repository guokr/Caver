#!/usr/bin/env python
# encoding: utf-8

import torch
import os
# from .base import BaseModule
from .cnn import CNN
from .lstm import LSTM
from caver.config import *


MODEL_CLASS = {
    "CNN": CNN(ConfigCNN()),
    "LSTM": LSTM(ConfigLSTM())
}

class CaverModelInitiationTypeError(Exception):
    pass


class CaverModel(object):
    """
    Wrapper for models, make it simpler for inference
    """
    def __init__(self, path=None, device="cpu"):
        super().__init__()
        self._inside_model = None
        if path:
            self.load(path, device)


    def load(self, path, device):
        """ load model from file """
        # assert os.path.isfile(path)
        # self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        loaded_checkpoint = torch.load(os.path.join(path, "checkpoint_best.pt"),
                                       map_location=device)
        self.model_type = loaded_checkpoint["model_type"]
        self._inside_model = MODEL_CLASS[self.model_type]
        self._inside_model.load(loaded_checkpoint, path)
        self._inside_model.eval()


    def predict(self, batch_sequence_text, top_k=5):
        res = self._inside_model.predict(batch_sequence_text, top_k=top_k)
        return res


    def predict_prob(self, batch_sequence_text):
        batch_prob = self._inside_model.predict_prob(batch_sequence_text)
        return batch_prob.data.numpy()

