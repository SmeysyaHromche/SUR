# author: Myron Kukhta (xkukht01)

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

from .model import Model


class ImageBinaryClassifier(Model):
    """
    Classifier for image data. 
    Simple Linear SVC with Standard Scaler and PCA prefix.
    """

    def __init__(self, n_components=0.9, C=0.8):
        super().__init__()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("svm", LinearSVC(
                C=C,
                class_weight={0: 1, 1: 2},
                max_iter=10000,
                random_state=42
            ))
        ])

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.model.fit(X, y)

    def predict(self, X:np.ndarray, file_ids:np.ndarray|None=None):
        return self.model.predict(X)

    def score(self, X:np.ndarray, file_ids:np.ndarray|None=None):
        return self.model.decision_function(X)
    
    def validation(self):
        if self.X_dev is None or self.y_dev is None:
            return '', 0.0
        self.y_pred = self.predict(self.X_dev)
        _classification_report = classification_report(self.y_dev, self.y_pred)
        _f1_score = f1_score(self.y_dev, self.y_pred)
        self.log = f"{_classification_report} \nF1 : {_f1_score}"
        return _f1_score
    
    
    def self_load_weights(self, path_to_pkl:str):
        self.model = joblib.load(path_to_pkl)
