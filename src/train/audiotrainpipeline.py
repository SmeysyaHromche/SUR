from pathlib import Path

from .basetrainpipeline import BaseTrainPipeline
from .trainconfig import TrainConfig
from src.data import AudioDataset
from src.model import AudioBinaryClassifier

class AudioTrainPipeline(BaseTrainPipeline):

    def __init__(self, config:TrainConfig):
        super().__init__(config)

    def get_dataset_for_path(self, path) -> AudioDataset:
        return AudioDataset(path)
    
    def get_model(self) -> AudioBinaryClassifier:
        return AudioBinaryClassifier()
    
    def get_model_subtype(self) -> str:
        return "audio"

    def train(self):
        prefix_log = (
            "============================================================================\n"
            f"Audio Binary Classifier Model '{self.config.model_name}'\n"
            "============================================================================\n"
        )
        print(prefix_log)

        model, logs, f1_scores = super().train(self.folds)
        logs = prefix_log + logs
        
        avg_f1_score = sum(f1_scores) / len(self.folds)
        avg_f1_score_msg = f"Avg f1 score throw all folds: {avg_f1_score}"
        print(avg_f1_score_msg)
        logs = logs + avg_f1_score_msg
        if self.config.is_save_validation_log:
            self.store_log(logs, self.config.model_name)

        if model is not None and self.config.is_save_validation_log:
            model.self_store(self.config.out, self.config.model_name)
            