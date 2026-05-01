import joblib
from pathlib import Path

from .basepipeline import BasePipeline
from .trainconfig import TrainConfig
from src.data import AudioDataset
from src.model import Model

class AudioTrain(BasePipeline):

    def __init__(self, config:TrainConfig):
        self.config = config
        self.total_data_path = Path(self.config.data_path) / "total.csv"
        target_folds_data_dir = Path(self.config.data_path) / "target_folds"
        nontarget_folds_data_dir = Path(self.config.data_path) / "nontarget_folds"
        self.target_folds = self.prepare_folds(target_folds_data_dir)
        self.nontarget_folds = self.prepare_folds(nontarget_folds_data_dir)

    def get_dataset_for_path(self, path) -> AudioDataset:
        return AudioDataset(path)
    
    def get_model(self) -> Model:
        return None
    
    def get_model_subtype(self) -> str:
        return "audio"

    def train(self):
        print("================================")
        print("TARGET")
        print("================================")
        super().train(self.target_folds)
        
        print("================================")
        print("NON-TARGET")
        print("================================")
        super().train(self.nontarget_folds)
            