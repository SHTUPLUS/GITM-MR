import torch
import torch.nn as nn
import numpy as np
from detectron2.utils.refdet_basics import to_numpy, to_torch
from torch.nn.parameter import Parameter


class ScaleLayer(nn.Module):
    def __init__(self, init_value=20.0, init_bias=0.0, no_scale=False):
        super(ScaleLayer, self).__init__()
        self.scale = 1. if no_scale else Parameter(torch.tensor(init_value, dtype=torch.float32), requires_grad=True)
        self.bias = Parameter(torch.tensor(init_bias, dtype=torch.float32), requires_grad=True) if init_bias != 0 else None
        # self.reset_parameter(init_value, init_bias)

    def reset_parameter(self, init_value, init_bias):
        self.scale.data.fill_(init_value)
        if init_bias != 0:
            self.bias.data.fill_(init_bias)

    def forward(self, x):
        if self.bias:
            x = x + self.bias
        x = x.clone() * self.scale

        return x


class SoftmaxLoss(nn.Module):
    def __init__(self, init_scale=1., focal=0.0):
        super(SoftmaxLoss, self).__init__()
        self.scale_fun = ScaleLayer(init_scale)
        self.focal = focal

    def valid_mean(self, tens, valid):
        filtered = tens * valid
        summ = filtered.sum()
        avg = summ / valid.sum()
        return avg.item()

    def forward(self, score, cls, sent_gt, match=None):
        """

        :param score: bs,nbox
        :param cls: bs,nbox
        :param sent_gt: bs,
        :param match:
        :return:
        """
        bs, n = score.size(0), score.size(1)
        scaled_score = self.scale_fun(score)

        celoss = []
        logits = torch.zeros((bs, n), requires_grad=False).cuda()

        for i in range(bs):
            label = np.ones((n, ), dtype=np.float32)
            label[to_numpy((cls[i])) == -1.0] = 0
            valid_label = to_torch(label).cuda()
            raw_logits = score[i]
            probs = torch.exp(raw_logits) / torch.sum(torch.exp(raw_logits))
            scaled_logits = scaled_score[i]

            if match[i]:
                ce = -(scaled_logits[sent_gt[i].long()] - torch.log(torch.sum(torch.exp(scaled_logits) * valid_label) + 1e-20)) # to avoid underflow
                if self.focal > 0:
                    ce = ce * torch.pow((1 - probs[sent_gt[i]]), self.focal) * 0.25
                celoss.append(ce)

            logits[i, :] = raw_logits

        loss_dict = dict(loss=torch.stack(celoss).mean() if len(celoss)>0 else torch.tensor(0.).to(device=scaled_score.device))

        return loss_dict, logits