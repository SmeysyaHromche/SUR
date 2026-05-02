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
        _, logs, f1_scores = super().train(self.folds)
        
        avg_f1_score = sum(f1_scores) / len(self.folds)
        avg_f1_score_msg = f"Avg f1 score throw all folds: {avg_f1_score}"
        print(avg_f1_score_msg)
        logs = logs + avg_f1_score_msg

        if self.config.is_save_validation_log:
            log_path = Path(self.config.out) / "logs" / f"log_{self.get_model_subtype()}_model_{self.config.model_name}.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_log(log_path, logs)
    