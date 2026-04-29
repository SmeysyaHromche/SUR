from pathlib import Path
from .trainconfig import TrainConfig

class Train:

    def __init__(self):
        pass

    def train(self, config_path:str) -> None:
        config_path = Path(config_path)
        config = TrainConfig.model_validate_json(config_path.read_text())
        print(config)