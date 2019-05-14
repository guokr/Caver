#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import torch.nn.functional as F


class EnsembleException(Exception):
    pass


class EnsembleModel(object):

    """
    :param models: list of models， each model should have the same label number
    :type models: list
    :param model_ratio: each model`s ratio in weighted soft voting and if this para is empty means no weighted
    :type model_ratio: list

    For now, this only support soft voting methods.

    """

    def __init__(self, models, model_ratio):
        assert isinstance(models, list) and len(models) > 0
        self.model_consistance_checker(models)

        self.models = models
        self.labels = models[0]._inside_model.labels
        self.vocab = models[0]._inside_model.vocab
        self.epsilon = 1e-8

        self.methods = {
            "mean": self.mean,
            "log": self.log,
            "hmean": self.hmean,
            "gmean": self.gmean,
        }
        self.model_ratio = model_ratio

    def __str__(self):
        start = "====== ensemble summary =======\n"
        summary = "\n-------------\n".join([model.__str__() for model in self.models])
        return start + summary

    def model_consistance_checker(self, models):
        """
        check all models have same labels and vocab dict
        :param models: list of models
        :return:
        """
        for model in models:
            if model._inside_model.labels != models[0]._inside_model.labels:
                raise EnsembleException(
                    "all models in ensemble mode should have same labels and vocab dict"
                )
            if model._inside_model.vocab != models[0]._inside_model.vocab:
                raise EnsembleException(
                    "all models in ensemble mode should have same labels and vocab dict"
                )

    def mean(self, models_preds):
        """
        mean: arithmetic mean
        if self.model_ratio not empty, calculate weighted arithmetic mean
        :param models_preds: each model labels predict probability
        :return: ensemble probability for sentences
        """
        ensemble_batch_preds = torch.zeros(models_preds[0].shape)
        if len(self.model_ratio) == 0:
            for preds in models_preds:
                ensemble_batch_preds += preds
            ensemble_batch_preds = ensemble_batch_preds / len(self.models)
        else:
            for idx in range(len(models_preds)):
                preds = models_preds[idx]
                alpha = torch.full(preds.size(), self.model_ratio[idx])
                ensemble_batch_preds += alpha.mul(preds)

        return ensemble_batch_preds

    def log(self, preds):
        return np.exp(np.log(self.epsilon + preds).mean(axis=0))

    def hmean(self, models_preds):
        """
        hmean: harmonic mean
        :param models_preds:
        :return:
        """
        ensemble_batch_preds = torch.zeros(models_preds[0].shape)
        if len(self.model_ratio) == 0:
            for preds in models_preds:
                ensemble_batch_preds += 1 / preds
            ensemble_batch_preds = len(self.models) / ensemble_batch_preds
        else:
            for idx in range(len(models_preds)):
                preds = models_preds[idx]
                alpha = torch.full(preds.size(), self.model_ratio[idx])
                ensemble_batch_preds += alpha.mul(1 / preds)
            ensemble_batch_preds = 1 / ensemble_batch_preds
        return ensemble_batch_preds

    def gmean(self, models_preds):
        """
        gmean: geometric mean
        :param models_preds:
        :return:
        """
        ensemble_batch_preds = torch.ones(models_preds[0].shape)
        if len(self.model_ratio) == 0:
            for preds in models_preds:
                ensemble_batch_preds *= preds
            ensemble_batch_preds = ensemble_batch_preds ** (1 / len(self.models))
        else:
            for idx in range(len(models_preds)):
                preds = models_preds[idx]
                alpha = torch.full(preds.size(), self.model_ratio[idx])
                ensemble_batch_preds *= torch.pow(preds, alpha)
        return ensemble_batch_preds

    def _predict_text(self, batch_sequence_text, top_k, method):
        """
        for each sentence in batch, fisrt get it label probability for each model, then ensemble by soft voting and model ratio, finally get ensemble probability
        soft voting include ['mean', 'hmean', 'gmean']
        weighted soft voting when self.model_ratio is not empty

        :param batch_sequence_text: list of sentences
        :param top_k: top_k lables we want
        :param method: soft voting method
        :return:
        """
        models_preds = [
            model._inside_model._get_model_output(
                batch_sequence_text=batch_sequence_text,
                vocab_dict=self.vocab,
                device="cpu",
            )
            for model in self.models
        ]
        models_preds_softmax = [F.softmax(preds, dim=1) for preds in models_preds]

        ensemble_batch_preds = self.methods[method](models_preds_softmax)

        batch_top_k_value, batch_top_k_index = torch.topk(
            torch.sigmoid(ensemble_batch_preds), k=top_k, dim=1
        )

        return batch_top_k_index

    def predict(self, batch_sequence_text, top_k, method):
        """

        :param batch_sequence_text: list of sentences
        :param top_k: top_k lables we want
        :param method: soft voting method
        :return:
        """
        batch_top_k_index = self._predict_text(batch_sequence_text, top_k, method)
        batch_top_k_index = batch_top_k_index.data.cpu().numpy()
        labels = []
        for pred in batch_top_k_index:
            labels.append([self.labels[idx] for idx in pred])
        return labels
