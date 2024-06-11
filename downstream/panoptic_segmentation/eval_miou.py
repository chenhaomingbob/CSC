# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py

import numpy as np


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    # hist: 行表示GT, 列表示 Pred
    # hist.sum(1)  按axis=1的轴相加, 每一行内元素进行相加, 相加结果为一个向量, 向量内的元素表示某个class的GT数量. = (TP + FN)
    # hist.sum(0)  按axis=0的轴相加, 每一列内元素进行相加, 相加结果为一个向量, 向量内的元素表示某个class的Pred数量. =  (TP + FP)
    # np.diag(hist)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  #


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 3)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


def fast_hist_for_sparse_sup(pred, label, n):
    data_for_bin_count = n * label.astype(int) + pred
    assert max(data_for_bin_count) <= n ** 2
    bin_count = np.bincount(data_for_bin_count, minlength=n ** 2)  #
    # the length of {bin_count} is equal to  np.amax(data_for_bin_count) +1
    return bin_count.reshape(n, n)


def fast_hist_crop_for_sparse_sup(output, target, unique_label, with_noise_label=True):
    nclass = len(unique_label)
    if with_noise_label:
        nclass += 1
    hist = fast_hist_for_sparse_sup(output.flatten(), target.flatten(), nclass)
    # 行表示GT；列表示Pred
    if with_noise_label:
        # hist[:, -1] = 0  # 将 Pred 为 noise的设置为0
        hist[-1, :] = 0  # 将GT 为 noise的设置为0 # 跟panoptic的混淆矩阵保持一致
    # if with_noise_label:
    #     hist = hist[:-1, :-1]  # [20,20] -> [19,19] 最后一行与最后一列均表示noise label.
    return hist
