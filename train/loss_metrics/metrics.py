# Modified from code by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import matplotlib.pyplot as plt

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def _fast_hist_dontcare(label_true, label_pred, dontcare_list, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    for dcc in dontcare_list:
        mask = mask & (label_true != dcc)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def _fast_hist_dontcare_weighted(label_true, label_pred, depth, dontcare_list, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    for dcc in dontcare_list:
        mask = mask & (label_true != dcc)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], weights=np.multiply(depth[mask],depth[mask]), minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall_Acc': acc,
            'Mean_Acc': acc_cls,
            'FreqW_Acc': fwavacc,
            'Mean_IoU': mean_iu,}, cls_iu


def scores_dontcare(label_trues, label_preds, n_class, dontcare_list):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist_dontcare(lt.flatten(), lp.flatten(), dontcare_list, n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    for dcc in dontcare_list:
        iu[dcc]=np.nan
    mean_iu = np.nanmean(iu)
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall_Acc': acc,
            'Acc_Cls': acc_cls,
            'Mean_IoU': mean_iu,}, cls_iu

def scores_dontcare_spatial(label_trues, label_preds, depths, n_class, dontcare_list):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for i in range(len(label_trues)):
        lt = label_trues[i]
        lp = label_preds[i]
        depth = depths[i]
        hist += _fast_hist_dontcare_weighted(lt.flatten(), lp.flatten(), depth.flatten(), dontcare_list, n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    for dcc in dontcare_list:
        iu[dcc]=np.nan
    mean_iu = np.nanmean(iu)
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall_Acc': acc,
            'Acc_Cls': acc_cls,
            'Mean_IoU': mean_iu,}, cls_iu


def scores_dontcare_nway(label_trues, label_preds, db_sizes, n_class, dontcare_list):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for j in range(len(label_trues)):
        label_trues_now=label_trues[j]
        label_preds_now=label_preds[j]
        pixel_weight=float(db_sizes[j][0])*db_sizes[j][0]/db_sizes[0][0]/db_sizes[0][0]
        for lt, lp in zip(label_trues_now, label_preds_now):
            hist += (_fast_hist_dontcare(lt.flatten(), lp.flatten(), dontcare_list, n_class)/pixel_weight)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    for dcc in dontcare_list:
        iu[dcc]=np.nan
    mean_iu = np.nanmean(iu)
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall_Acc': acc,
            'Acc_Cls': acc_cls,
            'Mean_IoU': mean_iu,}, cls_iu

def scores_dontcare_nway_weighted(label_trues, label_preds, depths, db_sizes, n_class, dontcare_list):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for j in range(len(label_trues)):
        label_trues_now=label_trues[j]
        label_preds_now=label_preds[j]
        depths_now=depths[j]
        pixel_weight=float(db_sizes[j][0])*db_sizes[j][0]/db_sizes[0][0]/db_sizes[0][0]
        for i in range(len(label_trues_now)):
            lt=label_trues_now[i]
            lp=label_preds_now[i]
            depth=depths_now[i]
            hist+=(_fast_hist_dontcare_weighted(lt.flatten(), lp.flatten(), depth.flatten(), dontcare_list, n_class)/pixel_weight)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    for dcc in dontcare_list:
        iu[dcc]=np.nan
    mean_iu = np.nanmean(iu)
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall_Acc': acc,
            'Acc_Cls': acc_cls,
            'Mean_IoU': mean_iu,}, cls_iu