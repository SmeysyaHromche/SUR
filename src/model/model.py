import numpy as np

from typing import Tuple

class Model:
    def __init__(self):
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
        self.file_ids_train = files_ids_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.file_ids_dev = files_ids_dev

        self.fit(self.X_train, self.y_train)

        if not with_validation or X_dev is None or y_dev is None:
            return ''
        validation_object = self.validation()
        
        return self.log, validation_object


    def fit(self, X:np.ndarray, y:np.ndarray):
        NotImplementedError("Implement me")
    
    def predict(self, X:np.ndarray):
        NotImplementedError("Implement me")
    
    def score(self, X:np.ndarray, y:np.ndarray):
        NotImplementedError("Implement me")

    def validation(self) -> Tuple[str, float]:
        NotImplementedError("Implement me")
