from pydantic import BaseModel

class TrainConfig(BaseModel):
    model: str
    total_data_csv_path: str
    folds_dir_path: str
    out: str
    model_name: str
    is_full_train: bool
    is_save_validation_log: bool