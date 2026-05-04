# author: Myron Kukhta (xkukht01)

from .productionconfig import ProductionConfig
from .baseproductionpipeline import BaseProductionPipeline

from src.data import ImageDataset
from src.model import ImageBinaryClassifier

class ImageProductionPipeline(BaseProductionPipeline):
    """
    Provide the logic for production pipeline of image binary classifier.
    """

    def __init__(self, config:ProductionConfig):
        super().__init__(config)
        self.image_dataset = ImageDataset(self.config.data_path)

    def get_model(self) -> ImageBinaryClassifier:
        model = ImageBinaryClassifier()
        model.self_load_weights(self.config.image_model_path)
        return model
    
    def get_data_for_evaluation(self):
        return ImageDataset(self.config.data_path).feature_extraction_for_evaluation()

