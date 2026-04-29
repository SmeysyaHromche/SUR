import cv2


class ImageHelper:

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def preprocessing(self, img_path: str):
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = self.clahe.apply(img)

        return img