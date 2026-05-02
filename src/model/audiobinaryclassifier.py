import numpy as np

from .model import Model
from .audiogmm import AudioGMM
from sklearn.metrics import classification_report, f1_score, confusion_matrix


class AudioBinaryClassifier(Model):

    def __init__(self, threshold: float = 1.5):
        self.gmm_target = AudioGMM(n_components_gmm=32)
        self.gmm_non_target = AudioGMM(n_components_gmm=32)
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_target = X[y == 1]
        X_non_target = X[y == 0]

        self.gmm_target.fit(X_target)
        self.gmm_non_target.fit(X_non_target)

    def score_samples(self, X: np.ndarray):
        """
        Returns likelihood-ratio score for each sample.

        score > 0  => more likely target
        score < 0  => more likely non-target
        """

        ll_target = self.gmm_target.score_samples(X)
        ll_non_target = self.gmm_non_target.score_samples(X)

        return ll_target - ll_non_target

    def predict(self, X: np.ndarray):
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(np.int64)

    def predict_proba(self, X: np.ndarray):

        scores = self.score_samples(X)

        prob_target = 1.0 / (1.0 + np.exp(-scores))
        prob_non_target = 1.0 - prob_target

        return np.vstack([prob_non_target, prob_target]).T

    def score(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


    def validation(self):
        # Split train/dev by labels
        target_train = self.X_train[self.y_train == 1]
        target_dev = self.X_dev[self.y_dev == 1]

        non_target_train = self.X_train[self.y_train == 0]
        non_target_dev = self.X_dev[self.y_dev == 0]

        # Prepare internal validation data for target GMM
        self.gmm_target.X_train = target_train
        self.gmm_target.X_dev = target_dev

        target_dev_ll = self.gmm_target.validation()
        target_gmm_log = self.gmm_target.log

        # Prepare internal validation data for non-target GMM
        self.gmm_non_target.X_train = non_target_train
        self.gmm_non_target.X_dev = non_target_dev

        non_target_dev_ll = self.gmm_non_target.validation()
        non_target_gmm_log = self.gmm_non_target.log

        # Binary classifier validation
        y_pred = self.predict(self.X_dev)
        scores = self.score_samples(self.X_dev)

        target_mask = self.y_dev == 1
        non_target_mask = self.y_dev == 0

        target_scores = scores[target_mask]
        non_target_scores = scores[non_target_mask]

        target_recall = (
            np.mean(y_pred[target_mask] == self.y_dev[target_mask])
            if np.any(target_mask)
            else 0.0
        )

        non_target_recall = (
            np.mean(y_pred[non_target_mask] == self.y_dev[non_target_mask])
            if np.any(non_target_mask)
            else 0.0
        )

        separation = (
            np.mean(target_scores) - np.mean(non_target_scores)
            if len(target_scores) and len(non_target_scores)
            else 0.0
        )

        cls_report = classification_report(
            self.y_dev,
            y_pred,
            labels=[0, 1],
            target_names=["non-target", "target"],
            zero_division=0
        )

        f1 = f1_score(
            self.y_dev,
            y_pred,
            zero_division=0
        )

        cm = confusion_matrix(
            self.y_dev,
            y_pred,
            labels=[0, 1]
        )

        self.log = (
            "================================\n"
            "TARGET GMM VALIDATION\n"
            "================================\n"
            f"{target_gmm_log}\n\n"

            "================================\n"
            "NON-TARGET GMM VALIDATION\n"
            "================================\n"
            f"{non_target_gmm_log}\n\n"

            "================================\n"
            "BINARY GMM CLASSIFIER VALIDATION\n"
            "================================\n\n"

            "Likelihood-ratio score:\n"
            "score = log p(x | target_gmm) - log p(x | non_target_gmm)\n"
            f"threshold = {self.threshold}\n\n"

            "TARGET SAMPLES\n"
            "--------------\n"
            f"count: {np.sum(target_mask)}\n"
            f"recall: {target_recall}\n"
            f"score mean: {np.mean(target_scores) if len(target_scores) else 0.0}\n"
            f"score std: {np.std(target_scores) if len(target_scores) else 0.0}\n"
            f"score min: {np.min(target_scores) if len(target_scores) else 0.0}\n"
            f"score max: {np.max(target_scores) if len(target_scores) else 0.0}\n\n"

            "NON-TARGET SAMPLES\n"
            "------------------\n"
            f"count: {np.sum(non_target_mask)}\n"
            f"recall: {non_target_recall}\n"
            f"score mean: {np.mean(non_target_scores) if len(non_target_scores) else 0.0}\n"
            f"score std: {np.std(non_target_scores) if len(non_target_scores) else 0.0}\n"
            f"score min: {np.min(non_target_scores) if len(non_target_scores) else 0.0}\n"
            f"score max: {np.max(non_target_scores) if len(non_target_scores) else 0.0}\n\n"

            "SEPARATION\n"
            "----------\n"
            f"target mean - non-target mean: {separation}\n"
            f"target GMM dev log-likelihood: {target_dev_ll}\n"
            f"non-target GMM dev log-likelihood: {non_target_dev_ll}\n\n"

            "CONFUSION MATRIX\n"
            "----------------\n"
            "labels order: [non-target, target]\n"
            f"{cm}\n\n"

            "CLASSIFICATION REPORT\n"
            "---------------------\n"
            f"{cls_report}\n"
            f"F1: {f1}"
        )

        return f1