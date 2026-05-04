# author: Myron Kukhta (xkukht01)

from pydantic import BaseModel

class ProductionConfig(BaseModel):
    """
    A scheme for configuration for production mode.
    """

    model: str
    data_path: str
    audio_target_model_path: str
    audio_non_targte_model_path: str
    image_model_path: str
    classification_out_path: str
