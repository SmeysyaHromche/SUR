# author: Myron Kukhta (xkukht01)

from pathlib import Path

from .audioproductionpipeline import AudioProductionPipeline
from .imageproductionpipeline import ImageProductionPipeline
from .productionconfig import ProductionConfig

class Production:
    """
    Entry point of production mode.
    """
    IMAGE = "image"
    AUDIO = "audio"
    
    def __init__(self, config_path:str):
        config_path = Path(config_path)
        self.config = ProductionConfig.model_validate_json(config_path.read_text())

    def get_pipeline(self):
        pipe_type = self.config.model
        if pipe_type == self.IMAGE:
            return ImageProductionPipeline(self.config)
        elif pipe_type == self.AUDIO:
            return AudioProductionPipeline(self.config)
        else:
            return None

    def run(self):
        pipeline = self.get_pipeline()
        pipeline.run()