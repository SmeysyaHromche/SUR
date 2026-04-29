import torch
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path

from .datasample import DataSample
from .surdataset import SurDataset
from .imagehelper import ImageHelper


class ImageDataset(SurDataset, Dataset):
    PNG_FORMAT = ".png"

    def __init__(self, meta_dataset: Path):
        super().__init__(meta_dataset)
        self.image_helper = ImageHelper()

    def get_expected_data_format(self) -> str:
        return self.PNG_FORMAT

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        img_features = self.image_helper.feature_extraction(str(img_path.resolve()), True)
        
        label = torch.tensor(label, dtype=torch.long)

        return DataSample(img_features, None, label)