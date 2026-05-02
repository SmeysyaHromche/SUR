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
        logs = ''
        for i, fold in enumerate(folds):
            field_log, validation_object = self.fold_train(i, fold)
            print(field_log)
            logs = logs + f"\n{field_log}"
            validation_objects.append(validation_object)
            break

        model = None
        # train model on full data and store it
        if self.config.is_full_train:
            X, y = self.feature_extraction_from_dataset(self.total_data_path, True)
            model = self.get_model()
            model.train(X, y)
            
            model_path = Path(self.config.out) / "models" / f"{self.get_model_subtype()}_model_{self.config.model_name}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model.model, model_path)

        return model, logs, validation_objects