import numpy as np

from pathlib import Path

from .surdataset import SurDataset
from .imagehelper import ImageHelper

class ImageDataset(SurDataset):
    PNG_FORMAT = ".png"

    def __init__(self, meta_dataset: Path):
        super().__init__(meta_dataset)
        self.image_helper = ImageHelper()

    def get_expected_data_format(self) -> str:
        return self.PNG_FORMAT

    def feature_extraction_from_dataset(self, is_train):
        X = []
        y = []
        file_ids = []

        for idx in range(len(self)):
            img_path, label = self.samples[idx]

            cnt_of_sample = 3 if label and is_train else 1

            for aug_idx in range(cnt_of_sample):

                img_features = self.image_helper.feature_extraction(
                    str(img_path.resolve()),
                    is_train
                )

                X.append(img_features)
                y.append(label)

                file_id = f"{idx}_{aug_idx}"
                file_ids.append(file_id)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        file_ids = np.asarray(file_ids)

        return X, y, file_ids