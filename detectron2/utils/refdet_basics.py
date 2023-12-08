# sgmn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import squareform, pdist

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


class NormalizeScale(nn.Module):

    def __init__(self, dim, init_norm=20):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        bottom_normalized = F.normalize(bottom, p=2, dim=-1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled


def max_overlap(b1, boxes):
    max_value = 0
    max_ind = -1
    ious = []
    for i in range(boxes.shape[0]):
        iou = compute_IoU(b1, boxes[i, :])
        ious.append(iou)
        if iou > max_value:
            max_value = iou
            max_ind = i
    return max_value, max_ind, ious


def compute_IoU(b1, b2):
    iw = min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1
    if iw <= 0:
        return 0
    ih = min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1
    if ih <= 0:
        return 0
    ua = float((b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1) + (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1) - iw*ih)
    return iw * ih / ua