import numpy as np

from .model import Model
from .audiogmm import AudioGMM
from sklearn.metrics import classification_report, f1_score


class AudioBinaryClassifier(Model):
    def __init__(self, threshold: float = -2.2):
        super().__init__()
        self.gmm_target = AudioGMM(n_components_gmm=32)
        self.gmm_non_target = AudioGMM(n_components_gmm=32)
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_target = X[y == 1]
        X_non_target = X[y == 0]

        self.gmm_target.fit(X_target)
        self.gmm_non_target.fit(X_non_target)

    def score_samples(self, X: np.ndarray):
        ll_target = self.gmm_target.score_samples(X)
        ll_non_target = self.gmm_non_target.score_samples(X)

        return ll_target - ll_non_target
    
    def score(self, X: np.ndarray, file_ids:np.ndarray|None=None):
        frame_scores = self.score_samples(X)

        if file_ids is None:
            return frame_scores

        file_scores = []
        unique_file_ids = np.array(list(dict.fromkeys(file_ids)))

        for file_id in unique_file_ids:
            file_mask = file_ids == file_id
            file_score = np.mean(frame_scores[file_mask])
            file_scores.append(file_score)

        return np.asarray(file_scores, dtype=np.float32)

    def predict(self, X: np.ndarray, file_ids:np.ndarray|None=None):
        scores = self.score(X, file_ids)
        return (scores > self.threshold).astype(np.int64)

    def find_best_threshold(self, scores, y_true):
        best_threshold = None
        best_f1 = -1.0

        thresholds = np.linspace(scores.min(), scores.max(), 500)

        for threshold in thresholds:
            y_pred = (scores > threshold).astype(np.int64)

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_f1

    def validation(self):
        # Split train/dev by frame labels
        target_train = self.X_train[self.y_train == 1]
        target_dev = self.X_dev[self.y_dev == 1]

        non_target_train = self.X_train[self.y_train == 0]
        non_target_dev = self.X_dev[self.y_dev == 0]

        # Target GMM validation
        self.gmm_target.X_train = target_train
        self.gmm_target.X_dev = target_dev

        target_dev_ll = self.gmm_target.validation()
        target_gmm_log = self.gmm_target.log

        # Non-target GMM validation
        self.gmm_non_target.X_train = non_target_train
        self.gmm_non_target.X_dev = non_target_dev

        non_target_dev_ll = self.gmm_non_target.validation()
        non_target_gmm_log = self.gmm_non_target.log

        # file level validation

        unique_file_ids = np.array(
            list(dict.fromkeys(self.files_ids_dev))
        )

        scores = self.score(self.X_dev, self.files_ids_dev)
        y_pred = self.predict(self.X_dev, self.files_ids_dev)

        file_labels = []

        for file_id in unique_file_ids:
            file_mask = self.files_ids_dev == file_id
            file_labels.append(self.y_dev[file_mask][0])

        y_true = np.asarray(file_labels, dtype=np.int64)

        # metrics

        target_mask = y_true == 1
        non_target_mask = y_true == 0

        target_scores = scores[target_mask]
        non_target_scores = scores[non_target_mask]

        target_recall = (
            np.mean(y_pred[target_mask] == y_true[target_mask])
            if np.any(target_mask)
            else 0.0
        )

        non_target_recall = (
            np.mean(y_pred[non_target_mask] == y_true[non_target_mask])
            if np.any(non_target_mask)
            else 0.0
        )

        separation = (
            np.mean(target_scores) - np.mean(non_target_scores)
            if len(target_scores) and len(non_target_scores)
            else 0.0
        )

        cls_report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["non-target", "target"],
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        # log

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
            "BINARY GMM CLASSIFIER VALIDATION FILE\n"
            "================================\n\n"
            f"threshold: {self.threshold}\n\n"

            "TARGET FILES\n"
            "------------\n"
            f"count: {np.sum(target_mask)}\n"
            f"recall: {target_recall}\n"
            f"score mean: {np.mean(target_scores) if len(target_scores) else 0.0}\n"
            f"score std: {np.std(target_scores) if len(target_scores) else 0.0}\n"
            f"score min: {np.min(target_scores) if len(target_scores) else 0.0}\n"
            f"score max: {np.max(target_scores) if len(target_scores) else 0.0}\n\n"

            "NON-TARGET FILES\n"
            "----------------\n"
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

            "CLASSIFICATION REPORT\n"
            "---------------------\n"
            f"{cls_report}\n"
            f"F1: {f1}"
        )

        return f1
    
    def self_store(self, out_path:str, model_name:str):
        self.store_model(self.gmm_target, out_path, model_name + "_target")
        self.store_model(self.gmm_non_target, out_path, model_name + "_nontarget")
    
    def self_load_weights(self, path_to_pkl:str):
        pass

    def self_load_gmm_weights(self, path_to_target_pkl:str, path_to_nontarget_pkl:str):
        self.gmm_target.self_load_weights(path_to_target_pkl)
        self.gmm_non_target.self_load_weights(path_to_nontarget_pkl)
