import cv2
import numpy as np

from skimage.feature import hog


class ImageHelper:

    def __init__(self, scaler=None):
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        self.scaler = scaler

    def _augment(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape

        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-10, 10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        noise = np.random.normal(0, 5, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def _hog(self, img):
        """
        Histogram of Oriented Gradients

        Calculates the gradient for each pixel along the X and Y axes.

        The image is divided into NxM cells, and each cell stores how many gradients point in different orientations.

        Then groups of cells are taken and normalized.

        Finally, flatten is applied to all of this, producing a one-dimensional array of gradients.
        Args:
            img: input image
        Returns:
            1d HOG representation array
        """        
        return hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False
        )


    def feature_extraction(self, img_path: str, is_augmentation: bool):
        img = cv2.imread(img_path)  # [H, W, BGR]

        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        # preprocessing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = self.clahe.apply(img)

        if is_augmentation:
            img = self._augment(img)

        # feature extraction
        img_to_hog = self._hog(img)  # [F]
        
        return img_to_hog
