# author: Myron Kukhta (xkukht01)

import numpy as np
import joblib

from typing import Tuple
from pathlib import Path

class Model:
    """
    Base class for classifier model.
    """
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
        self.files_ids_train = None
        self.X_dev = None
        self.y_dev = None
        self.y_pred = None
        self.files_ids_dev = None
        self.log = ''

    def train(
            self, 
            X_train:np.ndarray, 
            y_train:np.ndarray, 
            files_ids_train: np.ndarray,
            X_dev:np.ndarray|None=None, 
            y_dev:np.ndarray|None=None, 
            files_ids_dev: np.ndarray|None=None,
            with_validation:bool=False
        ) -> Tuple[str, float]:
        self.X_train = X_train
        self.y_train = y_train
        self.files_ids_train = files_ids_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.files_ids_dev = files_ids_dev

        self.fit(self.X_train, self.y_train)

        if not with_validation or X_dev is None or y_dev is None:
            return ''
        validation_object = self.validation()
        
        return self.log, validation_object

    def store_model(self, model, out_path:str, model_name:str) -> None:
        if model is None:
            return
        model_path = Path(out_path) / "models" / f"{model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f'Model successfuly saved on path: {model_path}')

    def fit(self, X:np.ndarray, y:np.ndarray):
        NotImplementedError("Implement me")
    
    def predict(self, X:np.ndarray, file_ids:np.ndarray|None=None):
        NotImplementedError("Implement me")
    
    def score(self, X:np.ndarray, file_ids:np.ndarray|None=None):
        NotImplementedError("Implement me")

    def validation(self) -> Tuple[str, float]:
        NotImplementedError("Implement me")

    def self_store(self, out_path:str, model_name:str):
        self.store_model(self.model, out_path, model_name)

    def self_load_weights(self, path_to_pkl:str):
        raise NotImplementedError("Implement me")
