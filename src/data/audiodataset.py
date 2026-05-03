import numpy as np

from pathlib import Path

from .surdataset import SurDataset
from .audiohelper import AudioHelper

class AudioDataset(SurDataset):
    WAV_FORMAT = ".wav"

    def __init__(self, meta_dataset: Path):
        super().__init__(meta_dataset)
        self.audio_helper = AudioHelper()

    def get_expected_data_format(self) -> str:
        return self.WAV_FORMAT

    def feature_extraction_from_dataset(self, is_train):
        X = []
        y = []
        file_ids = []

        for idx, row in self.df.iterrows():
            audio_path = Path(row[0])
            label = int(row[1])
            
            if not self.is_valid_data_format(audio_path):
                continue
            
            features = self.audio_helper.feature_extraction(
                str(audio_path.resolve()),
                is_augmentation=False
            )

            if len(features) == 0:
                continue

            file_id = f"{idx}"

            X.extend(features)
            y.extend([label] * len(features))
            file_ids.extend([file_id] * len(features))

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        file_ids = np.asarray(file_ids)

        return X, y, file_ids
    
    def feature_extraction_for_evaluation(self):
        X = []
        file_ids = []

        for _, row in self.df.iterrows():
            audio_path = Path(row[0])

            if not self.is_valid_data_format(audio_path):
                continue

            audio_features = self.audio_helper.feature_extraction(
                str(audio_path.resolve()),
                False
            )

            X.extend(audio_features)
            file_ids.extend([audio_path.stem] * len(audio_features))

        X = np.asarray(X, dtype=np.float32)
        file_ids = np.asarray(file_ids)

        return X, file_ids