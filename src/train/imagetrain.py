import joblib
from pathlib import Path
from typing import List

from .basepipeline import BasePipeline
from .trainconfig import TrainConfig
from src.model import ImageBinaryClassifier
from src.data import ImageDataset

class ImageTrain(BasePipeline):

    def __init__(self, config:TrainConfig):
        self.config = config
        self.total_data_path = Path(self.config.data_path) / "total.csv"
        folds_data_dir = Path(self.config.data_path) / "folds"
        self.folds = self.prepare_folds(folds_data_dir)

    def get_dataset_for_path(self, path) -> ImageDataset:
        return ImageDataset(path)
    
    def get_model(self) -> ImageBinaryClassifier:
        return ImageBinaryClassifier()
    
    def get_model_subtype(self) -> str:
        return "img"

    def train(self):
        super().train(self.folds)
            