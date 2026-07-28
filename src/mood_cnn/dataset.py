"""Mood image dataset: loads raw images + mood labels from mood_labels.csv
(columns: image_path, label, provenance — provenance is 'human' or 'pseudo').
"""
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MoodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_names: list[str], input_size: int, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.input_size = input_size
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))

        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1, :]  # horizontal flip

        img_tensor = torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1)).float() / 255.0
        img_tensor = self.normalize(img_tensor)
        label = self.class_to_idx[row["label"]]
        return img_tensor, label
