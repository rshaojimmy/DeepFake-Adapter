import json
import os
import os.path
import random

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (
    Compose,
    FancyPCA,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    HueSaturationValue,
    ImageCompression,
    OneOf,
    RandomBrightnessContrast,
    ShiftScaleRotate,
    ToGray,
)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_data(file):
    info = pd.read_csv(file)
    try:
        img_list = info["image"].tolist()
    except Exception:
        img_list = info["catimage"].tolist()
    label_list = info["label"].tolist()
    return img_list, label_list


def make_dataset(csv_file):
    dataset = []

    imgs, labels = read_data(csv_file)

    for i in range(len(imgs)):
        dataset.append((imgs[i], labels[i]))

    return dataset


def create_train_transforms(image_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def create_val_transforms(image_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def img_to_tensor(im, normalize=None):
    import torchvision.transforms.functional as F

    tensor = torch.from_numpy(np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def create_albu_train_transforms(image_size):
    return Compose(
        [
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            albumentations.augmentations.geometric.resize.Resize(image_size, image_size),
        ]
    )


class DeepFakeClassifierDataset(Dataset):
    def __init__(
        self,
        args,
        cfg,
        data_path=None,
        mode="train",
    ):
        super().__init__()
        self.data_root = data_path
        self.cfg = cfg
        self.mode = mode
        self.label_smoothing = 0.01

        self.albu = args.albu

        if self.mode.startswith("train"):
            if args.dataset_split == "alltype":
                self.data = make_dataset(
                    os.path.join(data_path, args.dataset_name, "image_lists", mode, "alltype.csv")
                )
            else:
                self.data = make_dataset(
                    os.path.join(
                        data_path,
                        args.dataset_name,
                        "image_lists",
                        f"{mode}",
                        f"{args.dataset_split}.csv",
                    )
                )

            if self.albu:
                self.transforms = create_albu_train_transforms(cfg["img_size"])
            else:
                self.transforms = create_train_transforms(
                    cfg["img_size"], cfg["normalize"]["mean"], cfg["normalize"]["std"]
                )

        elif self.mode.startswith("validation") or self.mode.startswith("test"):
            if args.dataset_split == "alltype":
                self.data = make_dataset(
                    os.path.join(data_path, args.dataset_name, "image_lists", mode, "alltype.csv")
                )
            else:
                self.data = make_dataset(
                    os.path.join(
                        data_path,
                        args.dataset_name,
                        "image_lists",
                        mode,
                        f"{args.dataset_split}.csv",
                    )
                )

            self.transforms = create_val_transforms(cfg["img_size"], cfg["normalize"]["mean"], cfg["normalize"]["std"])

        random.shuffle(self.data)

    def __getitem__(self, index: int):
        img_path, label = self.data[index]

        if self.albu and self.mode.startswith("train"):
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data = self.transforms(image=image)
            image = data["image"]

            image = img_to_tensor(image, self.cfg["normalize"])
        else:
            image = Image.open(img_path).convert("RGB")
            if self.transforms:
                image = self.transforms(image)

        return image, np.array((label,))

    def __len__(self):
        return len(self.data)


class DeepFakeClassifierDataset_test(Dataset):
    def __init__(self, args, cfg, test_dataset_name):
        super().__init__()
        self.data_root = args.data_dir
        self.test_level = args.test_level
        if "_" in args.results_path:
            dataset_suffix = args.results_path.split("_")[-1]
        self.test_dataset_name = test_dataset_name
        if args.test_level == "video":
            if test_dataset_name.startswith("FaceForensicspp"):
                json_file = os.path.join(
                    self.data_root,
                    test_dataset_name,
                    f"image_lists/test/{args.test_dataset_split}_vid.json",
                )

            elif test_dataset_name in ["DFDC", "Celeb-DF", "DeeperForensics-1.0"]:
                json_file = os.path.join(
                    self.data_root,
                    test_dataset_name,
                    f"image_lists/{test_dataset_name}_vid_balance_vid.json",
                )

            self.data = json.load(open(json_file))

        elif args.test_level == "frame":
            if test_dataset_name.startswith("FaceForensicspp"):
                self.data = make_dataset(
                    f"{self.data_root}/{test_dataset_name}/image_lists/test/{args.test_dataset_split}.csv"
                )
            elif test_dataset_name in ["Celeb-DF", "DeeperForensics-1.0"]:
                self.data = make_dataset(
                    f"{self.data_root}/{test_dataset_name}/image_lists/{test_dataset_name}_vid_balance_test_MTCNN_align.csv"
                )
            elif test_dataset_name in ["DFDC"]:
                self.data = make_dataset(
                    f"{self.data_root}/{test_dataset_name}/image_lists/{test_dataset_name}_vid_balance_test_largemargin.csv"
                )

        self.transforms = create_val_transforms(cfg["img_size"], cfg["normalize"]["mean"], cfg["normalize"]["std"])

    def __getitem__(self, index: int):
        if self.test_level == "video":
            vid_data = self.data[index]
            img = []
            vid_dir = vid_data["video_dir"]
            vid_frm = vid_data["video_frame"]
            label = vid_data["video_label"]

            for frm in vid_frm:
                img_path = f"{vid_dir}/{frm}"

                image = Image.open(img_path).convert("RGB")
                if self.transforms:
                    image = self.transforms(image)

                img.append(image)

        elif self.test_level == "frame":
            img_path, label = self.data[index]
            image = Image.open(img_path).convert("RGB")
            if self.transforms:
                img = self.transforms(image)

        return img, np.array((label,))

    def __len__(self):
        return len(self.data)
