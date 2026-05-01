import numpy as np

from typing import Tuple
from sklearn.metrics import classification_report, f1_score

class Model:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.y_pred = None
        self.log = ''

    def train(self, X_train:np.ndarray, y_train:np.ndarray, X_dev:np.ndarray|None=None, y_dev:np.ndarray|None=None, with_validation:bool=False) -> Tuple[str, float]:
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev

        self.fit(self.X_train, self.y_train)

        if not with_validation or X_dev is None or y_dev is None:
            return ''
        
        self.y_pred = self.predict(self.X_dev)
        _classification_report = classification_report(self.y_dev, self.y_pred)
        _f1_score = f1_score(self.y_dev, self.y_pred)
        self.log = f"{_classification_report} \nF1 : {_f1_score}"
        
        return self.log, _f1_score


    def fit(self, X:np.ndarray, y:np.ndarray):
        NotImplementedError("Implement me")
    
    def predict(self, X:np.ndarray):
        NotImplementedError("Implement me")
    
    def score(self, X:np.ndarray, y:np.ndarray):
        NotImplementedError("Implement me")