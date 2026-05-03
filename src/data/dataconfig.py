from pydantic import BaseModel

class DataConfig(BaseModel):
    source_data_dirs: list[str]
    output_data_dir: str
    trg_person_id: str
    with_folds: bool