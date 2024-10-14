import math

import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR, StepLR
from torch.optim.rmsprop import RMSprop

from tools.schedulers import ExponentialLRScheduler, PolyLR


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import json
from typing import List, Optional

import torch.distributed as dist
from torch import Tensor


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_optimizer(optimizer_config, model, sum_steps, master_params=None):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    if optimizer_config.get("classifier_lr", -1) != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if k.find("encoder") != -1:
                net_params.append(v)
            else:
                classifier_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        if master_params:
            params = master_params
        else:
            params = model.parameters()

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            params,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"],
            nesterov=optimizer_config["nesterov"],
        )
    # elif optimizer_config["type"] == "FusedSGD":
    #     optimizer = FusedSGD(params,
    #                          lr=optimizer_config["learning_rate"],
    #                          momentum=optimizer_config["momentum"],
    #                          weight_decay=optimizer_config["weight_decay"],
    #                          nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(
            params, lr=optimizer_config["learning_rate"], weight_decay=optimizer_config["weight_decay"]
        )
    # elif optimizer_config["type"] == "FusedAdam":
    #     optimizer = FusedAdam(params,
    #                           lr=optimizer_config["learning_rate"],
    #                           weight_decay=optimizer_config["weight_decay"])
    # elif optimizer_config["type"] == "AdamW":
    #     optimizer = AdamW(params,
    #                            lr=optimizer_config["learning_rate"],
    #                            weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "RmsProp":
        optimizer = RMSprop(
            params, lr=optimizer_config["learning_rate"], weight_decay=optimizer_config["weight_decay"]
        )
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        # scheduler = LRStepScheduler(optimizer, **optimizer_config["schedule"]["params"])
        scheduler = StepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "clr":
        scheduler = CyclicLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = ExponentialLRScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "poly":
        scheduler = PolyLR(optimizer, max_iter=sum_steps)
        # scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":

        def linear_lr(it):
            return (
                it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]
            )

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler


def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out


def nested_tensor_from_tensor_list(imgsize, tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = [3, imgsize, imgsize]
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.cuda(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.cuda(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError("Weight format not supported by conversion.")
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= 3 / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    # _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode="bicubic", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def load_weights(model, checkpoint_path: str, prefix: str = ""):
    """Load weights from .npz checkpoints for official Google Brain Flax implementation"""
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])))
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(_n2p(w[f"{bp}conv{r + 1}/kernel"]))
                        getattr(block, f"norm{r + 1}").weight.copy_(_n2p(w[f"{bp}gn{r + 1}/scale"]))
                        getattr(block, f"norm{r + 1}").bias.copy_(_n2p(w[f"{bp}gn{r + 1}/bias"]))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f"{bp}conv_proj/kernel"]))
                        block.downsample.norm.weight.copy_(_n2p(w[f"{bp}gn_proj/scale"]))
                        block.downsample.norm.bias.copy_(_n2p(w[f"{bp}gn_proj/bias"]))
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1], _n2p(w[f"{prefix}embedding/kernel"]))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.weight.requires_grad = False

    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    model.patch_embed.proj.bias.requires_grad = False

    model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    model.cls_token.requires_grad = False

    pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, "num_tokens", 1), model.patch_embed.grid_size
        )
    model.pos_embed.copy_(pos_embed_w)
    model.pos_embed.requires_grad = False

    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.weight.requires_grad = False

    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    model.norm.bias.requires_grad = False

    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]:
        model.head.weight.copy_(_n2p(w[f"{prefix}head/kernel"]))
        model.head.weight.requires_grad = False

        model.head.bias.copy_(_n2p(w[f"{prefix}head/bias"]))
        model.head.bias.requires_grad = False
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.weight.requires_grad = False

        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.norm1.bias.requires_grad = False

        block.attn.qkv.weight.copy_(
            torch.cat([_n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T for n in ("query", "key", "value")])
        )
        block.attn.qkv.weight.requires_grad = False

        block.attn.qkv.bias.copy_(
            torch.cat([_n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1) for n in ("query", "key", "value")])
        )
        block.attn.qkv.bias.requires_grad = False

        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        block.attn.proj.weight.requires_grad = False

        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        block.attn.proj.bias.requires_grad = False

        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(_n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"]))
            getattr(block.mlp, f"fc{r + 1}").weight.requires_grad = False

            getattr(block.mlp, f"fc{r + 1}").bias.copy_(_n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"]))
            getattr(block.mlp, f"fc{r + 1}").bias.requires_grad = False

        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.weight.requires_grad = False

        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))
        block.norm2.bias.requires_grad = False
