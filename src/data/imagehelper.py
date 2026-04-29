import cv2
import numpy as np
import torch

from skimage.feature import hog
from skimage.feature import local_binary_pattern


class ImageHelper:

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def _augment(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape

        # rotation
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # contrats (alpha) and brightness (beta)
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-10, 10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # gaussian noise
        noise = np.random.normal(0, 5, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img
    
    def _hog(self, img):        
        return hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False
        )
    
    def _lbp(self, img):
        radius = 1
        n_points = 8 * radius

        return local_binary_pattern(
            img,
            n_points,
            radius,
            method="uniform"
        )

    def feature_extraction(self, img_path: str, is_augmentation: bool):
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        # to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # to 128 x 128
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        # augmentation
        if is_augmentation:
            img = self._augment(img)
        # apply better contrast
        img = self.clahe.apply(img)

        # feature extraction
        img_to_hog = self._hog(img)
        img_to_lbp = self._lbp(img)

        img_features = np.concatenate([
            img_to_hog.ravel(),
            img_to_lbp.ravel()
        ])

        # concat features to one vec of data
        img_features = torch.from_numpy(img_features).float()

        return img_features