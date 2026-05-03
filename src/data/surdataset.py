import pandas as pd

from pathlib import Path

class SurDataset:

    def __init__(self, meta_dataset: Path):
        self.meta_dataset = meta_dataset

        self.df = pd.read_csv(self.meta_dataset, header=None, skiprows=1)

    def get_expected_data_format(self) -> str:
        """
        Example: ".png" or ".wav"
        """
        raise NotImplementedError("Implement me")
    
    def feature_extraction_from_dataset(self, is_train):
        raise NotImplementedError("Implement me")
    
    
    def feature_extraction_for_evaluation(self):
        raise NotImplementedError("Implement me")

    def is_valid_data_format(self, file: Path) -> bool:
        return file.is_file() and file.suffix == self.get_expected_data_format()

