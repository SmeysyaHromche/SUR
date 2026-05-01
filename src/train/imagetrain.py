import joblib
from pathlib import Path

from .basepipeline import BasePipeline
from .trainconfig import TrainConfig
from src.model import ImageBinaryClassifier

class ImageTrain(BasePipeline):

    def __init__(self, config:TrainConfig):
        self.config = config
        self.total_data_path = Path(self.config.data_path) / "total.csv"
        folds_data_dir = Path(self.config.data_path) / "folds"
        if not folds_data_dir.exists():
            raise FileNotFoundError(f"Folds directory not found: {folds_data_dir}")
        self.folds = sorted([d for d in folds_data_dir.iterdir() if d.is_dir()])

    def train(self):
        # cross validation
        f1_scores = []
        for i, fold in enumerate(self.folds):
            print("=====================")
            print(f"Fold: {i}")
            print("=====================")

            model = ImageBinaryClassifier()

            X_train, y_train, X_dev, y_dev = self.feature_extraction_from_fold_dataset(fold)

            log, f1_score = model.train(X_train, y_train, X_dev = X_dev, y_dev = y_dev, with_validation=True)
            f1_scores.append(f1_score)

            print(log)
            if self.config.is_save_validation_log:
                log_path = Path(self.config.out) / "logs" / f"log_img_model_{self.config.model_name}_fold_{i}.txt"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self.store_log(log_path, log)
        
        avg_f1_score = sum(f1_scores) / len(self.folds)
        print()
        print(f"Avg f1 score throw all folds: {avg_f1_score}")

        # train model on full data and store it
        if not self.config.is_full_train:
            return
        
        X, y = self.feature_extraction_from_dataset(self.total_data_path, True)
        model = ImageBinaryClassifier()
        model.train(X, y)
        
        model_path = Path(self.config.out) / "models" / f"img_model_{self.config.model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.model, model_path)
            