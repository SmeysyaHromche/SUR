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

        for idx in range(len(self)):
            audio_path, label = self.samples[idx]

            cnt_of_sample = 3 if label and is_train else 1

            for aug_idx in range(cnt_of_sample):

                features = self.audio_helper.feature_extraction(
                    str(audio_path.resolve()),
                    is_augmentation=(is_train and aug_idx > 0)
                )

                if len(features) == 0:
                    continue

                X.extend(features)

                y.extend([label] * len(features))

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        return X, y