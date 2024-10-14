"""
Copyright (c) 2018 Intel Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ------------ Online Hard Example mining ----------------------
class NLL_OHEM(th.nn.NLLLoss):
    """Online hard example mining.
    Needs input from nn.LogSotmax()"""

    def __init__(self, ratio):
        super(NLL_OHEM, self).__init__(None, True)
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = th.autograd.Variable(th.zeros(num_inst)).cuda()
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
        # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return th.nn.functional.nll_loss(x_hn, y_hn)


# ------------ Normal weights ----------------------


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


# ------------ AMSoftmax Loss ----------------------


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class AMSoftmaxLoss(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""

    margin_types = ["cos", "arc"]

    def __init__(self, margin_type="cos", gamma=0.0, m=0.5, s=30, t=1.0):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t

    def forward(self, cos_theta, target):
        if self.margin_type == "cos":
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m  # cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.0:
            return F.cross_entropy(self.s * output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t * cos_theta
            support_vecs_mask = (1 - index) * torch.lt(
                torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0
            )
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s * output, target)

        return focal_loss(F.cross_entropy(self.s * output, target, reduction="none"), self.gamma)


class AMSoftmax_OHEM(nn.Module):
    """Computes the AM-Softmax loss with cos or arc margin"""

    margin_types = ["cos", "arc"]

    def __init__(self, margin_type="cos", gamma=0.0, m=0.5, s=30, t=1.0, ratio=1.0):
        super(AMSoftmax_OHEM, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t
        self.ratio = ratio

    def forward(self, cos_theta, target):
        if self.margin_type == "cos":
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m  # cos(theta+m)
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        # ------- online hard example mining --------------------
        def get_subidx(self, x, y, ratio):
            num_inst = x.size(0)
            num_hns = int(ratio * num_inst)
            x_ = x.clone()
            inst_losses = th.autograd.Variable(th.zeros(num_inst)).cuda()

            for idx, label in enumerate(y.data):
                inst_losses[idx] = -x_.data[idx, label]

            _, idxs = inst_losses.topk(num_hns)
            return idxs

        out = F.log_softmax(output, dim=1)
        idxs = self.get_subidx(out, target, self.ratio)  # select hard examples

        output2 = output.index_select(0, idxs)
        target2 = target.index_select(0, idxs)

        if self.gamma == 0 and self.t == 1.0:
            return F.cross_entropy(self.s * output2, target2)

        if self.t > 1:
            h_theta = self.t - 1 + self.t * cos_theta
            support_vecs_mask = (1 - index) * torch.lt(
                torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0
            )
            output2 = torch.where(support_vecs_mask, h_theta, output2)
            return F.cross_entropy(self.s * output2, target2)

        return focal_loss(F.cross_entropy(self.s * output2, target2, reduction="none"), self.gamma)
