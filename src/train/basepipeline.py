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


    def store_log(self, log:str, model_name:str) -> None:
        log_path =  Path(self.config.out) / "logs" / f"log_{self.get_model_subtype()}_model_{model_name}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(log)
            print(f'Log of training successfuly saved on path: {log_path}')

    def store_model(self, model, model_name) -> None:
        model_path = Path(self.config.out) / "models" / f"{self.get_model_subtype()}_model_{model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.model, model_path)
        print(f'Model successfuly saved on path: {model_path}')

    def feature_extraction_from_dataset(self, path:Path, is_train:bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_dataset = self.get_dataset_for_path(path)
        X, y, files_ids = train_dataset.feature_extraction_from_dataset(is_train)
        return X, y, files_ids
    
    def feature_extraction_from_fold_dataset(self, fold:Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        fold_train_data = fold / "train.csv"
        X_train, y_train, files_ids_train = self.feature_extraction_from_dataset(fold_train_data, True)

        fold_dev_data = fold / "dev.csv"
        X_dev, y_dev, files_ids_dev = self.feature_extraction_from_dataset(fold_dev_data, False)

        return X_train, y_train, files_ids_train, X_dev, y_dev, files_ids_dev
    
    def fold_train(self, i:int, fold:Path):
        field_log = f"\n=====================\nFold: {i}\n====================="

        model = self.get_model()

        X_train, y_train, files_ids_train, X_dev, y_dev, files_ids_dev = self.feature_extraction_from_fold_dataset(fold)

        log, validation_object = model.train(
            X_train=X_train, 
            y_train=y_train, 
            files_ids_train=files_ids_train,
            X_dev = X_dev, 
            y_dev = y_dev, 
            files_ids_dev=files_ids_dev,
            with_validation=True
        )
        field_log = field_log + f"\n{log}"
        return field_log, validation_object

    def train(self, folds:List[Path]):
        # cross validation
        validation_objects = []
        logs = (
            "=====================================\n"
            "Cross Validation throw folds\n"
            "====================================="
        )
        print(logs)

        for i, fold in enumerate(folds):
            field_log, validation_object = self.fold_train(i, fold)
            print(field_log)
            logs = logs + f"\n{field_log}"
            validation_objects.append(validation_object)

        model = None
        model_full_train_log = (
            "=====================================\n"
            "Model Train on full data: "
        )
        model_full_train_status = ''
        
        # train model on full data
        if self.config.is_full_train:
            X, y, files_ids_train = self.feature_extraction_from_dataset(self.total_data_path, True)
            model = self.get_model()
            model.train(X, y, files_ids_train)
            model_full_train_status = 'successful'
        else:
            model_full_train_status = 'skiped'
        
        model_full_train_log = model_full_train_log + model_full_train_status + "\n=====================================\n"
        print(model_full_train_log)
        logs = logs + model_full_train_log

        return model, logs, validation_objects