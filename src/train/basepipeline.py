import joblib
import numpy as np

from pathlib import Path
from typing import Tuple, List

from src.data import SurDataset
from src.model import Model

class BasePipeline:

    def get_dataset_for_path(self, path) -> SurDataset:
        raise NotImplementedError("Implement me")
    
    def get_model(self) -> Model:
        raise NotImplementedError("Implement me")
    
    def get_model_subtype(self) -> str:
        raise NotImplementedError("Implement me")
    

    def prepare_folds(self, folds_dir:Path) -> List[Path]:
        if not folds_dir.exists():
            raise FileNotFoundError(f"Folds directory not found: {folds_dir}")
        return sorted([d for d in folds_dir.iterdir() if d.is_dir()])


    def store_log(self, path, txt:str) -> None:
        with open(path, "w") as f:
            f.write(txt)

    def feature_extraction_from_dataset(self, path:Path, is_train:bool) -> Tuple[np.ndarray, np.ndarray]:
        train_dataset = self.get_dataset_for_path(path)
        X, y = train_dataset.feature_extraction_from_dataset(is_train)
        return X, y
    
    def feature_extraction_from_fold_dataset(self, fold:Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fold_train_data = fold / "train.csv"
        X_train, y_train = self.feature_extraction_from_dataset(fold_train_data, True)

        fold_dev_data = fold / "dev.csv"
        X_dev, y_dev = self.feature_extraction_from_dataset(fold_dev_data, False)

        return X_train, y_train, X_dev, y_dev
    
    def train(self, folds:List[Path]):
        # cross validation
        f1_scores = []
        for i, fold in enumerate(folds):
            print("=====================")
            print(f"Fold: {i}")
            print("=====================")

            model = self.get_model()

            X_train, y_train, X_dev, y_dev = self.feature_extraction_from_fold_dataset(fold)
            continue

            log, f1_score = model.train(X_train, y_train, X_dev = X_dev, y_dev = y_dev, with_validation=True)
            f1_scores.append(f1_score)

            print(log)
            if self.config.is_save_validation_log:
                log_path = Path(self.config.out) / "logs" / f"log_{self.get_model_subtype()}_model_{self.config.model_name}_fold_{i}.txt"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self.store_log(log_path, log)
        
        avg_f1_score = sum(f1_scores) / len(folds)
        print()
        print(f"Avg f1 score throw all folds: {avg_f1_score}")

        # train model on full data and store it
        if not self.config.is_full_train:
            return
        
        X, y = self.feature_extraction_from_dataset(self.total_data_path, True)
        model = self.get_model()
        model.train(X, y)
        
        model_path = Path(self.config.out) / "models" / f"{self.get_model_subtype()}_model_{self.config.model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.model, model_path)