import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from .model import Model


class AudioGMM(Model):

    def __init__(
        self,
        n_components_gmm = 8,
        covariance_type="diag"
    ):
        self.model = Pipeline([
            ("scaler", StandardScaler()),

            ("gmm", GaussianMixture(
                n_components=n_components_gmm,
                covariance_type=covariance_type,
                reg_covar=1e-5,
                n_init=5,
                max_iter=200,
                random_state=42
            ))
        ])

    def fit(self, X: np.ndarray, y=None):
        self.model.fit(X)

    def predict(self, X: np.ndarray):
        # tell me from which gaussian cloud it is
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray):
        # probability for each cloud
        return self.model.predict_proba(X)

    def score_samples(self, X: np.ndarray):
        # log-likelihood for each x
        return self.model.score_samples(X)

    def score(self, X: np.ndarray, y=None):
        # mean log-likelihood
        return self.model.score(X)
    
    def validation(self):
        train_ll = self.score(self.X_train)
        dev_ll = self.score(self.X_dev)

        train_scores = self.score_samples(self.X_train)
        dev_scores = self.score_samples(self.X_dev)

        self.log = (
            f"Train avg log-likelihood: {train_ll}\n"
            f"Dev avg log-likelihood: {dev_ll}\n\n"
            f"Train score mean: {np.mean(train_scores)}\n"
            f"Train score std: {np.std(train_scores)}\n"
            f"Train score min: {np.min(train_scores)}\n"
            f"Train score max: {np.max(train_scores)}\n\n"
            f"Dev score mean: {np.mean(dev_scores)}\n"
            f"Dev score std: {np.std(dev_scores)}\n"
            f"Dev score min: {np.min(dev_scores)}\n"
            f"Dev score max: {np.max(dev_scores)}"
        )

        return dev_ll