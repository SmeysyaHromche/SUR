import numpy as np

from pathlib import Path
from typing import Tuple

from src.data import ImageDataset

class BasePipeline:

    def __call__(self):
        pass

    def store_log(self, path, txt:str) -> None:
        with open(path, "w") as f:
            f.write(txt)

    def feature_extraction_from_dataset(self, path:Path, is_train:bool) -> Tuple[np.ndarray, np.ndarray]:
        train_dataset = ImageDataset(path)
        X, y = train_dataset.feature_extarction_from_dataset(is_train)
        return X, y
    
    def feature_extraction_from_fold_dataset(self, fold:Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fold_train_data = fold / "train.csv"
        X_train, y_train = self.feature_extraction_from_dataset(fold_train_data, True)

        fold_dev_data = fold / "dev.csv"
        X_dev, y_dev = self.feature_extraction_from_dataset(fold_dev_data, False)

        return X_train, y_train, X_dev, y_dev