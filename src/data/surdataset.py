import pandas as pd
import torch

from pathlib import Path

from .datasample import DataSample

class SurDataset:

    def __init__(self, meta_dataset: Path):
        self.meta_dataset = meta_dataset

        self.df = pd.read_csv(self.meta_dataset, header=None)

        self.samples = self._filter_valid_files()
        self.len = len(self.samples)

    def get_expected_data_format(self) -> str:
        """
        Example: ".png" or ".wav"
        """
        raise NotImplementedError("Implement me")

    def is_valid_data_format(self, file: Path) -> bool:
        return file.is_file() and file.suffix == self.get_expected_data_format()

    def _filter_valid_files(self):
        valid = []

        for _, row in self.df.iterrows():
            path_str = row[0]
            label = row[1]

            file_path = Path(path_str)

            if self.is_valid_data_format(file_path):
                label = int(label)
                valid.append((file_path, label))

        return valid

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return path, label
    
    @staticmethod
    def collate_datasamples(batch):
        images = None
        if batch[0].image is not None:
            images = torch.stack([sample.image for sample in batch])
        
        audios = None
        if batch[0].audio is not None:
            audios = torch.stack([sample.audio for sample in batch])
        labels = torch.stack([sample.label for sample in batch])

        return DataSample(images, audios, labels)