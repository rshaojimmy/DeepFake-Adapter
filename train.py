import os
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import math
import random
from types import MethodType

import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from easydict import EasyDict
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
from timm.models import create_model
from timm.models.layers import trunc_normal_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models  # noqa
import util.misc as misc
from datasets.dataset import DeepFakeClassifierDataset, DeepFakeClassifierDataset_test
from tools.config import load_config
from tools.env import init_dist
from tools.metrics import Metrics
from tools.utils import AverageMeter, load_weights
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, auc):
        self.info("{set}-{idx:d} epoch | auc:{auc:.4f}%".format(set=set, idx=idx, auc=auc))

    logger.epochInfo = MethodType(epochInfo, logger)

    def epochInfo_two_metrics(self, set, idx, auc, acc):
        self.info("{set}-{idx:d} epoch | auc:{auc:.4f}% | acc:{acc:.4f}%".format(set=set, idx=idx, auc=auc, acc=acc))

    logger.epochInfo_two_metrics = MethodType(epochInfo_two_metrics, logger)

    def epochInfo_three_metrics(self, set, idx, auc, acc, eer):
        self.info(
            "{set}-{idx:d} epoch | auc:{auc:.4f}% | acc:{acc:.4f}%| eer:{eer:.4f}%".format(
                set=set, idx=idx, auc=auc, acc=acc, eer=eer
            )
        )

    logger.epochInfo_three_metrics = MethodType(epochInfo_three_metrics, logger)

    return logger


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


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def construct_optimizer(model, args, cfg):
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "bn" in name:
                bn_params.append(p)
            else:
                non_bn_parameters.append(p)
    optim_params = [
        {"params": bn_params, "weight_decay": 0.0},
        {"params": non_bn_parameters, "weight_decay": cfg["optimizer"]["weight_decay"]},
    ]
    if cfg["optimizer"]["type"] == "SGD":
        return torch.optim.SGD(
            optim_params,
            lr=args.lr,
            weight_decay=cfg["optimizer"]["weight_decay"],
            momentum=cfg["optimizer"]["momentum"],
        )
    elif cfg["optimizer"]["type"] == "Adam":
        # optim_params = ([p for p in model.parameters()])
        return optim.Adam(optim_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)


def evalute(val_dataloader, model):
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


def evalute_metric(val_dataloader, model):
    # switch model to evaluation mode
    metric = Metrics()
    model.eval()

    with torch.no_grad():
        for steps, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            imgs, labels = imgs.cuda(), labels.float().cuda()
            cls_score = model(imgs)

            metric.update(labels.detach(), cls_score.detach())

            # break
        mm = metric.get_mean_metrics()
        AUC = mm[2]

    return AUC


def train(
    args,
    cfg,
    train_dataloader,
    train_sampler,
    val_dataloader,
    model,
    summary_writer,
    logger,
    log_dir,
):
    model_without_ddp = model.module

    if cfg["optimizer"]["type"] == "SGD":
        optimizer = torch.optim.SGD(
            [p for name, p in model.named_parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=cfg["optimizer"]["weight_decay"],
            momentum=cfg["optimizer"]["momentum"],
        )
    if cfg["optimizer"]["type"] == "Adam":
        optimizer = optim.Adam(
            [p for name, p in model.named_parameters() if p.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0005,
        )

    if args.log:
        logger.info(optimizer)
    loss_scaler = NativeScaler()

    if args.resume:
        pretrain_dir_split = args.resume.split("/")[:-1]
        best_model_dir = os.path.join("/".join(pretrain_dir_split), "best_model.pt")

        logger.info(f"Loading best model from {best_model_dir}...")
        checkpoint = torch.load(best_model_dir, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["best_state_dict"])
        best_val_auc = checkpoint["best_val_auc"]

        logger.info(f"best_val_auc is  {best_val_auc}...")
    else:
        if args.cross_dataset or args.cross_type_online_test:
            best_val_auc_dict = {}
            best_val_acc_dict = {}
            best_val_eer_dict = {}
            for val_dataloader_name in args.test_dataset_name_list:
                best_val_auc_dict[val_dataloader_name] = 0
                best_val_acc_dict[val_dataloader_name] = 0
                best_val_eer_dict[val_dataloader_name] = 0
        else:
            best_val_auc = 0
            best_val_acc = 0
            best_val_eer = 0

    max_epochs = cfg["optimizer"]["epochs"]

    # criterion = nn.BCEWithLogitsLoss()
    if cfg["optimizer"]["criterion"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    if cfg["optimizer"]["criterion"] == "AMSoftmaxLoss":
        criterion = AMSoftmaxLoss(gamma=0.0, m=0.45, s=30, t=1.0)
    if args.log:
        print("criterion = %s" % str(criterion))

    global_step = args.start_epoch * len(train_dataloader)
    if args.log:
        logger.info(f"global_step: {global_step}...")

    if cfg["optimizer"]["scheduler"] == "StepLR":
        lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif cfg["optimizer"]["scheduler"] == "cosine":
        import util.lr_sched as lr_sched

    for current_epoch in range(args.start_epoch, max_epochs):
        train_sampler.set_epoch(current_epoch)
        loss_logger = AverageMeter()
        # ----------
        #  Training
        # ----------
        current_epoch_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        if args.log:
            logger.info(f"############# Starting Epoch {current_epoch} | LR: {current_epoch_lr} #############")

        model.train()
        criterion.train()

        if args.log:
            train_dataloader = tqdm(train_dataloader, dynamic_ncols=True)

        optimizer.zero_grad()
        for steps, (images, labels) in enumerate(train_dataloader):
            if cfg["optimizer"]["scheduler"] == "cosine":
                lr_sched.adjust_learning_rate(optimizer, steps / len(train_dataloader) + current_epoch, args, cfg)

            current_lr = optimizer.state_dict()["param_groups"][0]["lr"]

            # images, labels = images.cuda(), labels.float().cuda()
            images, labels = images.cuda(), labels.squeeze(1).long().cuda()

            if images.shape[0] < 2:
                continue

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            if not math.isfinite(loss):
                if args.log:
                    print(f"Loss is {loss}, stopping training")
                sys.exit(1)

            loss_scaler(
                loss,
                optimizer,
                clip_grad=None,
                parameters=model.parameters(),
                create_graph=False,
            )

            optimizer.zero_grad()

            torch.cuda.synchronize()

            loss_logger.update(loss.item(), images.size(0))
            global_step += 1

            # ============ tensorboard train log info ============#
            if args.log:
                lossinfo = {
                    "lr": current_lr,
                    "Train_Loss": loss.item(),
                    "Train_Loss_avg": loss_logger.avg,
                }
                for tag, value in lossinfo.items():
                    summary_writer.add_scalar(tag, value, global_step)

                # ============ print the train log info ============#
                train_dataloader.set_description(
                    "lr: {lr:.8f} | loss: {loss:.8f} ".format(loss=loss_logger.avg, lr=current_lr)
                )
            # break

        # ============ train model save ============#
        if current_epoch % args.model_save_epoch == 0:
            if args.log:
                model_save_path = os.path.join(log_dir, "snapshots")
                mkdir(model_save_path)
                torch.save(
                    {
                        "epoch": current_epoch,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": loss_scaler.state_dict(),
                    },
                    os.path.join(model_save_path, "model-{}.pt".format(current_epoch)),
                )

        # ----------
        #  Validation
        # ----------
        if args.cross_dataset or args.cross_type_online_test:
            if current_epoch % args.val_epoch == 0:
                if args.log:
                    model_save_path = os.path.join(log_dir, "snapshots")
                    mkdir(model_save_path)

                    for val_dataloader_name in val_dataloader:
                        val_dataloader_data = val_dataloader[val_dataloader_name]

                        AUC, ACC, EER = evalute(val_dataloader_data, model.module)
                        # AUC = evalute_metric(val_dataloader_data, model.module)

                        # ============ print the val log info ============#
                        # logger.epochInfo(f'Validation of {val_dataloader_name}', current_epoch, AUC*100)
                        logger.epochInfo_three_metrics(
                            f"Current Validation of {val_dataloader_name}",
                            current_epoch,
                            AUC * 100,
                            ACC,
                            EER * 100,
                        )
                        # ============ tensorboard val log info ============#
                        valinfo = {
                            f"Val_AUC_{val_dataloader_name}": 100 * AUC,
                        }
                        for tag, value in valinfo.items():
                            summary_writer.add_scalar(tag, value, current_epoch)

                        if AUC >= best_val_auc_dict[val_dataloader_name]:
                            best_val_auc_dict[val_dataloader_name] = AUC
                            best_val_acc_dict[val_dataloader_name] = ACC
                            best_val_eer_dict[val_dataloader_name] = EER
                            torch.save(
                                {
                                    "best_val_auc": best_val_auc_dict[val_dataloader_name],
                                    "best_state_dict": model.module.state_dict(),
                                },
                                os.path.join(
                                    model_save_path,
                                    f"best_model_{val_dataloader_name}.pt",
                                ),
                            )

                        # logger.epochInfo(f'!!!!!!!!!!Best Validation of {val_dataloader_name}', current_epoch, best_val_auc_dict[val_dataloader_name])
                        logger.epochInfo_three_metrics(
                            f"!!!!!!!!!!Best Validation of {val_dataloader_name}",
                            current_epoch,
                            best_val_auc_dict[val_dataloader_name] * 100,
                            best_val_acc_dict[val_dataloader_name],
                            best_val_eer_dict[val_dataloader_name] * 100,
                        )
        else:
            if current_epoch % args.val_epoch == 0:
                if args.log:
                    model_save_path = os.path.join(log_dir, "snapshots")
                    mkdir(model_save_path)
                    AUC, ACC, EER = evalute(val_dataloader, model.module)

                    # ============ print the val log info ============#
                    # logger.epochInfo('Validation', current_epoch, AUC*100)
                    # logger.epochInfo_two_metrics('Validation', current_epoch, AUC*100, ACC)
                    logger.epochInfo_three_metrics(
                        "Validation",
                        current_epoch,
                        AUC * 100,
                        ACC,
                        EER * 100,
                    )
                    # ============ tensorboard val log info ============#
                    valinfo = {
                        "Val_AUC": 100 * AUC,
                        "Val_ACC": 100 * ACC,
                    }
                    for tag, value in valinfo.items():
                        summary_writer.add_scalar(tag, value, current_epoch)

                    if AUC >= best_val_auc:
                        best_val_auc = AUC
                        best_val_acc = ACC
                        best_val_eer = EER
                        torch.save(
                            {
                                "best_val_auc": best_val_auc,
                                "best_state_dict": model.module.state_dict(),
                            },
                            os.path.join(model_save_path, "best_model.pt"),
                        )

                    # logger.epochInfo_two_metrics(f'!!!!!!!!!!Best Validation AUC and ACC are:', current_epoch, best_val_auc*100, best_val_acc)
                    logger.epochInfo_three_metrics(
                        "Best Validation",
                        current_epoch,
                        best_val_auc * 100,
                        best_val_acc,
                        best_val_eer * 100,
                    )

        if cfg["optimizer"]["scheduler"] == "StepLR":
            lr_sched.step()


def main_worker(gpu, args, cfg):
    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    if args.ffn_adapt:
        log_dir = os.path.join(
            args.results_path,
            args.dataset_name,
            args.dataset_split,
            cfg["network"],
            cfg["finetuning"]["method_name"],
            "log" + args.log_num,
        )
    else:
        log_dir = os.path.join(
            args.results_path,
            args.dataset_name,
            args.dataset_split,
            cfg["network"],
            "fine_tune",
            "log" + args.log_num,
        )
    # log_dir = os.path.join(args.results_path, args.dataset_name, args.dataset_split, cfg['network'], cfg['finetuning']['resnet_method_name']+'_'+cfg['finetuning']['method_name'], 'log'+ args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    logger = setlogger(log_file)

    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None

    if args.log:
        logger.info("******************************")
        logger.info(args)
        logger.info("******************************")
        logger.info(cfg)
        logger.info("******************************")

    # dataset
    batch_size = cfg["optimizer"]["batch_size"]

    train_dataset = DeepFakeClassifierDataset(args, cfg, mode="train", data_path=args.data_dir)
    if args.cross_dataset:
        val_dataset_dict = {}
        args.test_dataset_name_list = args.test_dataset_name[0].split(" ")
        for dataset_name in args.test_dataset_name_list:
            val_dataset = DeepFakeClassifierDataset_test(args, cfg, dataset_name)
            val_dataset_dict[dataset_name] = val_dataset
    elif args.cross_type_online_test:
        val_dataset_dict = {}
        args.test_dataset_name_list = args.test_dataset_name[0].split(" ")
        for dataset_name in args.test_dataset_name_list:
            args.dataset_split = dataset_name
            val_dataset = DeepFakeClassifierDataset(args, cfg, mode="test", data_path=args.data_dir)
            val_dataset_dict[dataset_name] = val_dataset
    else:
        val_dataset = DeepFakeClassifierDataset(args, cfg, mode="test", data_path=args.data_dir)

    if args.log:
        print("train:", len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,
        sampler=train_sampler,
    )

    if args.cross_dataset or args.cross_type_online_test:
        val_dataloader = {}
        for dataset_name in val_dataset_dict:
            val_dataset_one = val_dataset_dict[dataset_name]
            val_dataloader_one = torch.utils.data.DataLoader(
                val_dataset_one, batch_size=128, shuffle=True, num_workers=4
            )
            val_dataloader[dataset_name] = val_dataloader_one
            if args.log:
                print(f"validation of {dataset_name}:", len(val_dataset_one))
    else:
        if args.log:
            print("validation:", len(val_dataset))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

    # fine-tuning configs
    if cfg["finetuning"]["method_name"] == "bottleneck":
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
            ffn_num=cfg["finetuning"]["ffn_num"],
            d_model=cfg["finetuning"]["d_model"],
            ffn_adapter_drop=cfg["finetuning"]["drop"],
            use_learnable_pos_emb=cfg["finetuning"]["use_learnable_pos_emb"],
            # use_spatial_embed=cfg['finetuning']['use_spatial_embed'],
            num_heads_spatial_adapter=cfg["finetuning"]["num_heads_spatial_adapter"],
            interaction_indexes=cfg["finetuning"]["interaction_indexes"],
            h_size=cfg["finetuning"]["h_size"],
            w_size=cfg["finetuning"]["w_size"],
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
            ffn_num=cfg["finetuning"]["ffn_num"],
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
            ffn_adapter_drop=cfg["finetuning"]["drop"],
            scale=cfg["finetuning"]["scale"],
            depth_kernel=cfg["finetuning"]["depth_kernel"],
            conv_size=cfg["finetuning"]["conv_size"],
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
            attn_adapt=args.attn_adapt,
            ffn_adapt_method=cfg["finetuning"]["method_name"],
            ffn_option=cfg["finetuning"]["ffn_option"],
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar=cfg["finetuning"]["ffn_adapter_scalar"],
            d_model=cfg["finetuning"]["d_model"],
            with_mlp=cfg["finetuning"]["with_mlp"],
            scale=cfg["finetuning"]["scale"],
            h_size=cfg["finetuning"]["h_size"],
            w_size=cfg["finetuning"]["w_size"],
            use_learnable_pos_emb=cfg["finetuning"]["use_learnable_pos_emb"],
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
    elif cfg["finetuning"]["method_name"] == "LeFF_gfnet":
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=args.ffn_adapt,
            ffn_adapt_method=cfg["finetuning"]["method_name"],
            ffn_option=cfg["finetuning"]["ffn_option"],
            ffn_adapter_layernorm_option=cfg["finetuning"]["ffn_adapter_layernorm_option"],
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar=cfg["finetuning"]["ffn_adapter_scalar"],
            d_model=cfg["finetuning"]["d_model"],
            scale=cfg["finetuning"]["scale"],
            depth_kernel=cfg["finetuning"]["depth_kernel"],
            conv_size=cfg["finetuning"]["conv_size"],
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
        "LeFF_gfnet",
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

    if os.path.splitext(cfg["finetuning"]["pretrained_weights"])[-1].lower() in (".pth"):
        checkpoint = torch.load(cfg["finetuning"]["pretrained_weights"], map_location="cpu")

        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint
        logger.info(f"load pretrained checkpoint from {cfg['finetuning']['pretrained_weights']}")

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                if args.log:
                    print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        if args.log:
            print(msg)

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head,
        )

        # freeze all but the head
        for name, p in model.named_parameters():
            # print(name)

            if name in msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False if not args.fulltune else True
        for _, p in model.head.named_parameters():
            p.requires_grad = True
        # raise
    elif os.path.splitext(cfg["finetuning"]["pretrained_weights"])[-1].lower() in (".npz"):
        load_weights(model, cfg["finetuning"]["pretrained_weights"])
        if args.log:
            print(f"Load pretrained checkpoint from {cfg['finetuning']['pretrained_weights']}")

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head,
        )

        for _, p in model.head.named_parameters():
            p.requires_grad = True

    model.cuda()

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.log:
        # print("Model = %s" % str(model_without_ddp))
        logger.info("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = batch_size * misc.get_world_size()

    if cfg["optimizer"]["warmup_epochs"] == 0:
        args.lr = cfg["optimizer"]["blr"]
    else:
        args.lr = cfg["optimizer"]["blr"] * eff_batch_size / 256

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if "large" in cfg["network"]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    train(
        args,
        cfg,
        train_dataloader,
        train_sampler,
        val_dataloader,
        model,
        summary_writer,
        logger,
        log_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("--config", metavar="CONFIG_FILE", help="path to configuration file")
    arg("--results_path", type=str, default="results")
    arg("--data_dir", type=str, default=None)
    arg("--dataset_name", type=str, default=None)
    arg("--dataset_split", type=str, default=None)
    arg("--test_dataset_name", nargs="+", default=None)
    arg("--test_level", type=str, default="frame")
    arg("--csvfile", type=str, default=None)
    arg("--resume", type=str, default=None)
    arg("--log_num", "-l", type=str)

    arg("--model_save_epoch", type=int, default=1)
    arg("--val_epoch", type=int, default=1)
    arg("--manual_seed", type=int, default=777)

    arg("--rank", default=-1, type=int, help="node rank for distributed training")
    arg("--world_size", default=1, type=int, help="world size for distributed training")
    arg(
        "--dist-url",
        default="tcp://127.0.0.1:23458",
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

    arg("--start_epoch", default=0, type=int, metavar="N", help="start epoch")

    # AdaptFormer related parameters
    arg(
        "--ffn_adapt",
        default=False,
        action="store_true",
        help="whether activate AdaptFormer",
    )
    arg(
        "--attn_adapt",
        default=False,
        action="store_true",
        help="whether activate AdaptFormer",
    )
    arg(
        "--albu",
        default=False,
        action="store_true",
        help="whether activate AdaptFormer",
    )
    arg("--vpt", default=False, action="store_true", help="whether activate VPT")
    arg("--fulltune", default=False, action="store_true", help="full finetune model")
    arg(
        "--inception",
        default=False,
        action="store_true",
        help="whether use INCPETION mean and std (for Jx provided IN-21K pretrain",
    )
    arg("--cross_dataset", default=False, action="store_true")
    arg("--cross_type_online_test", default=False, action="store_true")

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    args = parser.parse_args()
    set_random_seed(args.manual_seed)
    cfg = load_config(args.config)

    if args.launcher == "none":
        args.launcher = "pytorch"
        main_worker(0, args, cfg)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, cfg))
