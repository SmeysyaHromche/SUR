# author: Myron Kukhta (xkukht01)

from .productionconfig import ProductionConfig
from .baseproductionpipeline import BaseProductionPipeline

from src.data import AudioDataset
from src.model import AudioBinaryClassifier

class AudioProductionPipeline(BaseProductionPipeline):
    """
    Provide the logic for production pipeline of audio binary classifier.
    """

    def __init__(self, config:ProductionConfig):
        super().__init__(config)
        self.image_dataset = AudioDataset(self.config.data_path)
    
    def get_model(self) -> AudioBinaryClassifier:
        model = AudioBinaryClassifier()
        model.self_load_gmm_weights(
            self.config.audio_target_model_path,
            self.config.audio_non_targte_model_path
        )
        return model
    
    def get_data_for_evaluation(self):
        return AudioDataset(self.config.data_path).feature_extraction_for_evaluation()
