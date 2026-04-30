from pydantic import BaseModel

class TrainConfig(BaseModel):
    model: str
    data_path: str
    epochs: int
    batch_size: int
    out: str