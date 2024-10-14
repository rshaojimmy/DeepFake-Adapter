import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import random
from types import MethodType

import numpy as np
import torch
from easydict import EasyDict
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
from timm.models import create_model
from tqdm import tqdm

import models  # noqa
from datasets.dataset import DeepFakeClassifierDataset_test
from models.xception_ff import xception
from tools.config import load_config
from tools.env import init_dist


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def test_auc(self, auc):
        self.info("auc:{auc:.4f}%".format(auc=auc))

    logger.test_auc = MethodType(test_auc, logger)

    def test_three_metrics(self, set, auc, acc, eer):
        self.info("{set} | auc:{auc:.4f}% | acc:{acc:.4f}%| eer:{eer:.4f}%".format(set=set, auc=auc, acc=acc, eer=eer))

    logger.test_three_metrics = MethodType(test_three_metrics, logger)

    return logger


def preset_model(args, cfg, model, logger):
    if cfg["network"].startswith("vit"):
        if args.ffn_adapt:
            checkpoint_dir = os.path.join(
                args.results_path,
                args.dataset_name,
                args.dataset_split,
                cfg["network"],
                cfg["finetuning"]["method_name"],
                args.log_num,
                "snapshots",
                args.ckpt,
            )
        else:
            checkpoint_dir = os.path.join(
                args.results_path,
                args.dataset_name,
                args.dataset_split,
                cfg["network"],
                "fine_tune",
                args.log_num,
                "snapshots",
                args.ckpt,
            )

    if cfg["network"].startswith("xception"):
        checkpoint_dir = os.path.join(
            args.results_path,
            args.dataset_name,
            args.dataset_split,
            cfg["network"],
            cfg["method_name"],
            args.log_num,
            "snapshots",
            args.ckpt,
        )

    checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    if args.ckpt == "best_model.pt":
        model.load_state_dict(checkpoint["best_state_dict"])
        model.cuda(args.gpu)
        best_val_acc = checkpoint["best_val_auc"]
        if args.log:
            logger.info(f"Loading model from {checkpoint_dir}...")
            logger.info(f"best_val_auc: {best_val_acc}...")
    else:
        model.load_state_dict(checkpoint["state_dict"])
        model.cuda(args.gpu)

        logger.info(f"Loading model from {checkpoint_dir}...")
    return model


def evalute_video(val_dataloader, model):
    # switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        y_true, y_pred = [], []

        for steps, (img_list, labels) in enumerate(tqdm(val_dataloader)):
            imgs = torch.cat(img_list, dim=0)
            labels = labels[0][0]

            imgs, labels = imgs.cuda(), labels.float().cuda()
            preds = model(imgs)
            score = torch.nn.functional.softmax(preds, dim=1)[:, 1].cpu().flatten().tolist()
            score_avg = sum(score) / len(score)
            y_pred.append(score_avg)
            y_true.append(labels.item())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        AUC = roc_auc_score(y_true, y_pred)

    return AUC


def evalute_frame(val_dataloader, model):
    # switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        y_true, y_pred = [], []

        for steps, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            imgs, labels = imgs.cuda(), labels.float().cuda()
            preds = model(imgs)
            y_pred.extend(torch.nn.functional.softmax(preds, dim=1)[:, 1].cpu().flatten().tolist())
            y_true.extend(labels.cpu().flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        AUC = roc_auc_score(y_true, y_pred)
    return AUC


def evalute(val_dataloader, model):  # -> tuple[Float, Any | float, tuple[Any, RootResults] | Any]:
    # switch model to evaluation mode
    model.eval()

    with torch.no_grad():
        y_true, y_pred = [], []
        nums_all = 0
        acc_all = 0

        for steps, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            imgs, labels = imgs.cuda(), labels.float().cuda()
            with torch.cuda.amp.autocast():
                preds = model(imgs)
            # y_pred.extend(preds.sigmoid().cpu().flatten().tolist())
            y_pred.extend(torch.nn.functional.softmax(preds, dim=1)[:, 1].cpu().flatten().tolist())
            y_true.extend(labels.cpu().flatten().tolist())

            pred_acc = preds.argmax(1)
            nums_all += labels.shape[0]
            acc_all += torch.sum(pred_acc == labels.squeeze(1))

            # break

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        AUC = roc_auc_score(y_true, y_pred)

        ACC = acc_all / nums_all

        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        EER = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return AUC, ACC, EER


def test(args, cfg, test_dataloader, model, logger):
    model = preset_model(args, cfg, model, logger)
    if args.test_level == "video":
        AUC = evalute_video(test_dataloader, model)
    if args.test_level == "frame":
        # AUC = evalute_frame(test_dataloader, model)
        AUC, ACC, EER = evalute(test_dataloader, model)
    # logger.test_auc(100*AUC)
    logger.test_three_metrics(
        "Validation",
        AUC * 100,
        ACC * 100,
        EER * 100,
    )


def main_worker(gpu, args, cfg):
    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    if cfg["network"].startswith("vit"):
        if args.ffn_adapt:
            log_dir = os.path.join(
                args.results_path,
                args.dataset_name,
                args.dataset_split,
                cfg["network"],
                cfg["finetuning"]["method_name"],
                args.log_num,
                "evaluation",
            )
        else:
            log_dir = os.path.join(
                args.results_path,
                args.dataset_name,
                args.dataset_split,
                cfg["network"],
                "fine_tune",
                args.log_num,
                "evaluation",
            )
    elif cfg["network"].startswith("xception"):
        log_dir = os.path.join(
            args.results_path,
            args.dataset_name,
            args.dataset_split,
            cfg["network"],
            cfg["method_name"],
            args.log_num,
            "evaluation",
        )

    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f"test_on_{args.test_dataset_name}_{args.test_dataset_split}_{args.test_level}.txt"
    log_file = os.path.join(log_dir, log_file_name)

    logger = setlogger(log_file)

    if args.log:
        logger.info("******************************")
        logger.info(args)
        logger.info("******************************")
        logger.info(cfg)
        logger.info("******************************")

    # dataset

    test_dataset = DeepFakeClassifierDataset_test(args, cfg, args.test_dataset_name)

    if args.log:
        print("Test:", len(test_dataset))
    if args.test_level == "video":
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    if args.test_level == "frame":
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    if cfg["network"].startswith("vit"):
        # fine-tuning configs
        if cfg["finetuning"]["method_name"] == "bottleneck":
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=args.ffn_adapt,
                attn_adapt=args.attn_adapt,
                ffn_adapt_method=cfg["finetuning"]["method_name"],
                ffn_option=cfg["finetuning"]["ffn_option"],
                ffn_adapter_layernorm_option=cfg["finetuning"]["ffn_adapter_layernorm_option"],
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar=cfg["finetuning"]["ffn_adapter_scalar"],
                ffn_num=cfg["finetuning"]["ffn_num"],
                d_model=cfg["finetuning"]["d_model"],
                ffn_adapter_drop=cfg["finetuning"]["drop"],
                use_learnable_pos_emb=cfg["finetuning"]["use_learnable_pos_emb"],
                # use_spatial_embed=cfg['finetuning']['use_spatial_embed'],
                num_heads_spatial_adapter=cfg["finetuning"]["num_heads_spatial_adapter"],
                interaction_indexes=cfg["finetuning"]["interaction_indexes"],
                # VPT related
                vpt_on=args.vpt,
                vpt_num=cfg["vpt_num"],
            )
        elif cfg["finetuning"]["method_name"] in ["convpass2", "convpass"]:
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=args.ffn_adapt,
                attn_adapt=args.attn_adapt,
                ffn_adapt_method=cfg["finetuning"]["method_name"],
                # ffn_resnet_adapt_method=cfg['finetuning']['resnet_method_name'],
                ffn_option=cfg["finetuning"]["ffn_option"],
                ffn_adapter_layernorm_option=cfg["finetuning"]["ffn_adapter_layernorm_option"],
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar=cfg["finetuning"]["ffn_adapter_scalar"],
                conv_dim=cfg["finetuning"]["conv_dim"],
                d_model=cfg["finetuning"]["d_model"],
                ffn_adapter_drop=cfg["finetuning"]["drop"],
                use_learnable_pos_emb=cfg["finetuning"]["use_learnable_pos_emb"],
                adapter_type=cfg["finetuning"]["adapter_type"],
                # use_spatial_embed=cfg['finetuning']['use_spatial_embed'],
                num_heads_spatial_adapter=cfg["finetuning"]["num_heads_spatial_adapter"],
                interaction_indexes=cfg["finetuning"]["interaction_indexes"],
                h_size=cfg["finetuning"]["h_size"],
                w_size=cfg["finetuning"]["w_size"],
                # VPT related
                vpt_on=args.vpt,
                vpt_num=cfg["vpt_num"],
            )
        elif cfg["finetuning"]["method_name"] == "LeFF":
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=args.ffn_adapt,
                attn_adapt=args.attn_adapt,
                ffn_adapt_method=cfg["finetuning"]["method_name"],
                ffn_option=cfg["finetuning"]["ffn_option"],
                ffn_adapter_layernorm_option=cfg["finetuning"]["ffn_adapter_layernorm_option"],
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar=cfg["finetuning"]["ffn_adapter_scalar"],
                d_model=cfg["finetuning"]["d_model"],
                scale=cfg["finetuning"]["scale"],
                depth_kernel=cfg["finetuning"]["depth_kernel"],
                conv_size=cfg["finetuning"]["conv_size"],
                ffn_adapter_drop=cfg["finetuning"]["drop"],
                use_learnable_pos_emb=cfg["finetuning"]["use_learnable_pos_emb"],
                # use_spatial_embed=cfg['finetuning']['use_spatial_embed'],
                num_heads_spatial_adapter=cfg["finetuning"]["num_heads_spatial_adapter"],
                interaction_indexes=cfg["finetuning"]["interaction_indexes"],
                # VPT related
                vpt_on=args.vpt,
                vpt_num=cfg["vpt_num"],
            )
        elif cfg["finetuning"]["method_name"] == "gfnet":
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=args.ffn_adapt,
                ffn_adapt_method=cfg["finetuning"]["method_name"],
                ffn_option=cfg["finetuning"]["ffn_option"],
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar=cfg["finetuning"]["ffn_adapter_scalar"],
                d_model=cfg["finetuning"]["d_model"],
                with_mlp=cfg["finetuning"]["with_mlp"],
                scale=cfg["finetuning"]["scale"],
                h_size=cfg["finetuning"]["h_size"],
                w_size=cfg["finetuning"]["w_size"],
                # VPT related
                vpt_on=args.vpt,
                vpt_num=cfg["vpt_num"],
            )
        elif cfg["finetuning"]["method_name"] == "bottleneck_gfnet":
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=args.ffn_adapt,
                ffn_adapt_method=cfg["finetuning"]["method_name"],
                ffn_adapter_layernorm_option=cfg["finetuning"]["ffn_adapter_layernorm_option"],
                ffn_option=cfg["finetuning"]["ffn_option"],
                ffn_adapter_init_option="lora",
                ffn_num=cfg["finetuning"]["ffn_num"],
                ffn_adapter_scalar=cfg["finetuning"]["ffn_adapter_scalar"],
                d_model=cfg["finetuning"]["d_model"],
                # with_mlp=cfg['finetuning']['with_mlp'],
                scale=cfg["finetuning"]["scale"],
                h_size=cfg["finetuning"]["h_size"],
                w_size=cfg["finetuning"]["w_size"],
                GFnet_pos=cfg["finetuning"]["GFnet_pos"],
                # VPT related
                vpt_on=args.vpt,
                vpt_num=cfg["vpt_num"],
            )

        # model
        if cfg["finetuning"]["method_name"] in [
            "bottleneck",
            "LeFF",
            "gfnet",
            "bottleneck_gfnet",
            "convpass",
            "convpass2",
        ]:
            model = create_model(
                cfg["network"],
                pretrained=False,
                img_size=cfg["img_size"],
                num_classes=cfg["nb_classes"],
                drop_rate=cfg["drop"],
                drop_path_rate=cfg["drop_path"],
                attn_drop_rate=cfg["attn_drop_rate"],
                global_pool=args.global_pool,
                tuning_config=tuning_config,
            )

        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head
        )

    elif cfg["network"].startswith("xception"):
        # model = TransferModel('xception', num_out_classes=cfg['nb_classes'], dropout=cfg['dropout'])
        model = xception(num_classes=cfg["nb_classes"])

    test(args, cfg, test_dataloader, model, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("--config", metavar="CONFIG_FILE", help="path to configuration file")
    arg("--results_path", type=str, default="results")
    arg("--test_level", type=str, default=None)
    arg("--data_dir", type=str, default=None)
    arg("--dataset_name", type=str, default=None)
    arg("--test_dataset_name", type=str, default=None)
    arg("--dataset_split", type=str, default=None)
    arg("--test_dataset_split", type=str, default=None)
    arg("--csvfile", type=str, default=None)
    arg("--resume", type=str, default=None)
    arg("--log_num", "-l", type=str)

    arg("--model_save_epoch", type=int, default=2)
    arg("--val_epoch", type=int, default=1)
    arg("--manual_seed", type=int, default=777)

    arg("--rank", default=-1, type=int, help="node rank for distributed training")
    arg("--world_size", default=1, type=int, help="world size for distributed training")
    arg(
        "--dist-url",
        default="tcp://127.0.0.1:23459",
        type=str,
        help="url used to set up distributed training",
    )
    arg("--dist-backend", default="nccl", type=str, help="distributed backend")
    arg(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    arg("--use_mean_pooling", default=True)

    arg("--start_epoch", default=0, type=int, metavar="N", help="start epoch")

    # AdaptFormer related parameters
    arg("--ffn_adapt", default=False, action="store_true", help="whether activate AdaptFormer")
    arg("--attn_adapt", default=False, action="store_true", help="whether activate AdaptFormer")
    arg("--vpt", default=False, action="store_true", help="whether activate VPT")
    arg("--fulltune", default=False, action="store_true", help="full finetune model")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument("--ckpt", default="best_model.pt", type=str)

    args = parser.parse_args()
    set_random_seed(args.manual_seed)
    cfg = load_config(args.config)

    main_worker(0, args, cfg)
