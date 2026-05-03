from pathlib import Path
from .trainconfig import TrainConfig
from .imagetrainpipeline import ImageTrainPipeline
from .audiotrainpipeline import AudioTrainPipeline

class Train:

    IMAGE = "image"
    AUDIO = "audio"

    def __init__(self, config_path:str):
        config_path = Path(config_path)
        self.config = TrainConfig.model_validate_json(config_path.read_text())
        
    def get_pipeline(self):
        pipe_type = self.config.model
        if pipe_type == self.IMAGE:
            return ImageTrainPipeline(self.config)
        elif pipe_type == self.AUDIO:
            return AudioTrainPipeline(self.config)
        else:
            return None

    def train(self) -> None:
        pipeline = self.get_pipeline()
        pipeline.train()