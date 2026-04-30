from pathlib import Path
from sklearn.metrics import classification_report, f1_score

from .trainconfig import TrainConfig
from src.data import ImageDataset
from src.model import ImageBinaryClassifier

class ImageTrain():

    def __init__(self, config:TrainConfig):
        self.config = config
        data_dir = Path(self.config.data_path)
        self.folds = [d for d in data_dir.iterdir() if d.is_dir()]

    def train(self):
        for i, fold in enumerate(self.folds):
            print("=====================")
            print(f"Session: {i}")
            print("=====================")

            model = ImageBinaryClassifier()

            fold_train_data = fold / "train.csv"
            train_dataset = ImageDataset(fold_train_data)
            X_train, y_train = train_dataset.feature_extarction_from_dataset(True)

            fold_dev_data = fold / "dev.csv"
            dev_dataset = ImageDataset(fold_dev_data)
            X_dev, y_dev = dev_dataset.feature_extarction_from_dataset(False)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_dev)

            print(classification_report(y_dev, y_pred))
            print("F1:", f1_score(y_dev, y_pred))
            
            