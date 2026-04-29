from pathlib import Path
from torch.utils.data import DataLoader

from src.data import ImageDataset
from .trainconfig import TrainConfig

class ImageTrain():

    def __init__(self, config:TrainConfig):
        self.config = config
        data_dir = Path(self.config.data_path)
        self.folds = [d for d in data_dir.iterdir() if d.is_dir()]
        
    def train(self):
        for fold in self.folds:
            fold_train_data = fold / "train.csv"
            dataset = ImageDataset(fold_train_data)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers= 4,
                collate_fn = dataset.collate_datasamples
            )
            i = 0
            for data_sample in dataloader:
                print(f'i{i}')
                print(data_sample.image)
                print(data_sample.audio)
                print(data_sample.label)
                i += 1
            break