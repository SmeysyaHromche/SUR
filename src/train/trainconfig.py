from pydantic import BaseModel

class TrainConfig(BaseModel):
    model: str
    data_path: str
    epochs: int
    batch_size: int
    out: str
    model_name: str
    is_full_train: bool
    is_save_validation_log: bool