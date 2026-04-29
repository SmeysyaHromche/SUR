from pathlib import Path
from .trainconfig import TrainConfig
from .imagetrain import ImageTrain

class Train:

    IMAGE = "image"
    AUDIO = "audio"

    def __init__(self, config_path:str):
        config_path = Path(config_path)
        self.config = TrainConfig.model_validate_json(config_path.read_text())
        
    def get_pipeline(self):
        # TODO: add all pipelines
        pipe_type = self.config.model
        if pipe_type == self.IMAGE:
            return ImageTrain(self.config)
        elif pipe_type == self.AUDIO:
            return None
        else:
            return None

    def train(self) -> None:
        print(self.config)
        pipeline = self.get_pipeline()
        pipeline.train()