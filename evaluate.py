#!/usr/bin/env python
# encoding: utf-8

import jieba
import csv
from tqdm import tqdm
import torch
import numpy as np



def compute_recall(array_pred, array_y):
    return float(len(np.intersect1d(array_pred, array_y)))/len(array_y)


def eval_recall(preds, target, top_k=5):
    # preds/target: batch_size x num_labels
    batch_size = preds.size(0)
    # print("origin shape..... ")
    # print(preds.shape)
    # print(target.shape)

    _, preds_idx = torch.topk(preds, k=5, dim=1)

    target_value, target_idx = torch.topk(target, k=5, dim=1)
    # preds_idx/target_idx: batch_size x top_k
    preds_idx_cpu = preds_idx.data.cpu().numpy()
    target_idx_cpu = target_idx.data.cpu().numpy()
    target_value = target_idx.data.cpu().numpy()


    batch_recall = 0.0

    for idx in range(batch_size):
        ground_truth_num = len(target_value.nonzero())
        _recall = compute_recall(preds_idx_cpu[idx], target_idx_cpu[idx][:ground_truth_num])
        batch_recall += _recall

    return batch_recall / batch_size




