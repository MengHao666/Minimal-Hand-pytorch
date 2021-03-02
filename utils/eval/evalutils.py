import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from utils.misc import *
from utils.heatmaputils import get_heatmap_pred


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_dists(preds, target, normalize, mask):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))  # (njoint, B)
    for b in range(preds.size(0)):
        for j in range(preds.size(1)):
            if mask[b][j] == 0:
                dists[j, b] = -1
            elif target[b, j, 0] < 1 or target[b, j, 1] < 1:
                dists[j, b] = -1
            else:
                dists[j, b] = torch.dist(preds[b, j, :], target[b, j, :]) / normalize[b]

    return dists


def dist_acc(dist, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def accuracy_heatmap(output, target, mask, thr=0.5):
    """ Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First to be returned is average accuracy across 'idxs', Second is individual accuracies
    """
    preds = get_heatmap_pred(output).float()  # (B, njoint, 2)
    gts = get_heatmap_pred(target).float()
    norm = torch.ones(preds.size(0)) * output.size(3) / 10.0  # (B, ), all 6.4:(1/10 of heatmap side)
    dists = calc_dists(preds, gts, norm, mask)  # (njoint, B)

    acc = torch.zeros(mask.size(1))
    avg_acc = 0
    cnt = 0

    for i in range(mask.size(1)):  # njoint
        acc[i] = dist_acc(dists[i], thr)
        if acc[i] >= 0:
            avg_acc += acc[i]
            cnt += 1

    if cnt != 0:
        avg_acc /= cnt

    return avg_acc, acc
