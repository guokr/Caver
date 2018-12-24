#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import torch.nn.functional as F


class EnsembleException(Exception):
    pass


class Ensemble(object):
    """
    :param list models: each model should have the same label number
    For now, this only support soft voting methods.

    """
    def __init__(self, models):
        assert isinstance(models, list) and len(models) > 0
        self.model_consistance_checker(models)

        self.models = models
        self.labels = models[0].labels
        self.vocab = models[0].vocab
        self.epsilon = 1e-8

        self.methods = {
            'mean': self.mean,
            'log': self.log,
            'hmean': self.hmean,
            'gmean': self.gmean,
        }


    def __str__(self):
        start = "====== ensemble summary =======\n"
        summary = "\n-------------\n".join([model.__str__() for model in self.models])
        return start+summary


    def model_consistance_checker(self, models):
        for model in models:
            if model.labels != models[0].labels:
                raise EnsembleException("all models in ensemble mode should have same labels and vocab dict")
            if model.vocab != models[0].vocab:
                raise EnsembleException("all models in ensemble mode should have same labels and vocab dict")



    def mean(self, models_preds):
        ensemble_batch_preds = torch.zeros(models_preds[0].shape)
        for preds in models_preds:
            # print(F.softmax(preds, dim=1))
            ensemble_batch_preds += preds
        ensemble_batch_preds = ensemble_batch_preds / len(self.models)
        # print(ensemble_res)
        return ensemble_batch_preds
        # return batch_top_k_index


    def log(self, preds):
        return np.exp(np.log(self.epsilon + preds).mean(axis=0))


    def hmean(self, models_preds):
        ensemble_batch_preds = torch.zeros(models_preds[0].shape)
        for preds in models_preds:
            # print(F.softmax(preds, dim=1))
            ensemble_batch_preds += 1/preds

        ensemble_batch_preds = len(self.models) / ensemble_batch_preds
        # print(ensemble_res)
        return ensemble_batch_preds


    def gmean(self, models_preds):
        ensemble_batch_preds = torch.ones(models_preds[0].shape)
        for preds in models_preds:
            # print(F.softmax(preds, dim=1))
            ensemble_batch_preds *= preds

        ensemble_batch_preds = ensemble_batch_preds**(1/len(self.models))
        # print(ensemble_res)
        return ensemble_batch_preds


    def _predict_text(self, batch_sequence_text, device="cpu", top_k=5, method='mean'):
        """
        :param str text: text
        :param str method: ['mean', 'hmean', 'gmean']

        mean: arithmetic mean
        hmean: harmonica mean
        gmean: geometric mean

        """
        models_preds = [model._predict_text(batch_sequence_text, self.vocab, device="cpu", top_k=5) for model in self.models]
        models_preds_softmax = [F.softmax(preds, dim=1) for preds in models_preds]

        # print("original models preds")
        # print(models_preds_softmax)

        ensemble_batch_preds = self.methods[method](models_preds_softmax)

        batch_top_k_value, batch_top_k_index = torch.topk(torch.sigmoid(ensemble_batch_preds), k=top_k, dim=1)

        return batch_top_k_index



    def predict(self, batch_sequence_text, top_k=5, method="mean"):
        batch_top_k_index = self._predict_text(batch_sequence_text, top_k=5, method="mean")
        batch_top_k_index = batch_top_k_index.data.cpu().numpy()
        labels = []
        for pred in batch_top_k_index:
            labels.append([self.labels[idx] for idx in pred])
        return labels
