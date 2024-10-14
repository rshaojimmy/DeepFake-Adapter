# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg['optimizer']['warmup_epochs']:
        lr = args.lr * epoch / cfg['optimizer']['warmup_epochs'] 
    else:
        lr = cfg['optimizer']['min_lr'] + (args.lr - cfg['optimizer']['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg['optimizer']['warmup_epochs']) / (cfg['optimizer']['epochs'] - cfg['optimizer']['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
