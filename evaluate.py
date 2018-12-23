#!/usr/bin/env python
# encoding: utf-8

import csv
from tqdm import tqdm
import torch
import numpy as np
import math



def compute_recall(array_pred, array_y):
    return float(len(np.intersect1d(array_pred, array_y)))/len(array_y)


def compute_precision(array_pred, array_y):
    correct_num = [0.0] * len(array_pred)
    predict_num = [1.0] * len(array_pred)

    for idx in range(len(correct_num)):
        if array_pred[idx] in array_y:
            correct_num[idx] = 1.0

    weighted_correct = 0.0
    weighted_predict = 0.0
    for idx in range(len(correct_num)):
        weighted_correct += correct_num[idx] / math.log(idx + 3.0)
        weighted_predict += predict_num[idx] / math.log(idx + 3.0)

    return weighted_correct / weighted_predict


def compute_f_score(recall_score, precision_score, f_type=1):
    """
    larget f_type if you prefer high precicion, in most cases, f1 is nice balance in recall and precision.
    """
    coef = 1+math.pow(f_type, 2)
    pr1 = precision_score + recall_score
    pr2 = math.pow(f_type, 2) * precision_score + recall_score
    if pr2 == 0:
        return 0
    return coef * pr1 / pr2


def evaluation(preds, target, top_k=5):
    # preds/target: batch_size x num_labels
    batch_size = preds.size(0)

    _, preds_idx = torch.topk(preds, k=5, dim=1)

    target_value, target_idx = torch.topk(target, k=5, dim=1)
    # preds_idx/target_idx: batch_size x top_k
    preds_idx_cpu = preds_idx.data.cpu().numpy()
    target_idx_cpu = target_idx.data.cpu().numpy()
    target_value = target_idx.data.cpu().numpy()

    batch_recall = 0.0
    batch_precision = 0.0
    batch_f_score = 0.0

    for idx in range(batch_size):
        ground_truth_num = len(target_value.nonzero())
        _recall = compute_recall(preds_idx_cpu[idx], target_idx_cpu[idx][:ground_truth_num])
        _precition = compute_precision(preds_idx_cpu[idx], target_idx_cpu[idx][:ground_truth_num])
        _f_score = compute_f_score(_recall, _precition)

        batch_recall += _recall
        batch_precision += _precition
        batch_f_score += _f_score

    # return the batch ave recall, precision and fscore at once

    return list(map(lambda x: x/batch_size, [batch_recall, batch_precision, batch_f_score]))
