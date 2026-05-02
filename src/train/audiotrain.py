from pathlib import Path

from .basepipeline import BasePipeline
from .trainconfig import TrainConfig
from src.data import AudioDataset
from src.model import AudioBinaryClassifier

class AudioTrain(BasePipeline):

    def __init__(self, config:TrainConfig):
        self.config = config
        self.total_data_path = Path(self.config.data_path) / "total.csv"
        folds_data_dir = Path(self.config.data_path) / "folds"
        self.folds = self.prepare_folds(folds_data_dir)

    def get_dataset_for_path(self, path) -> AudioDataset:
        return AudioDataset(path)
    
    def get_model(self) -> AudioBinaryClassifier:
        return AudioBinaryClassifier()
    
    def get_model_subtype(self) -> str:
        return "audio"

    def train(self):
        _, logs, f1_scores = super().train(self.folds)
        
        avg_f1_score = sum(f1_scores) / len(self.folds)
        avg_f1_score_msg = f"Avg f1 score throw all folds: {avg_f1_score}"
        #print(avg_f1_score_msg)
        logs = logs + avg_f1_score_msg

        if self.config.is_save_validation_log:
            log_path = Path(self.config.out) / "logs" / f"log_{self.get_model_subtype()}_model_{self.config.model_name}.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_log(log_path, logs)
            