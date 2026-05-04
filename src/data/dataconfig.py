# author: Myron Kukhta (xkukht01)

from pydantic import BaseModel

class DataConfig(BaseModel):
    """
    A scheme for configuration for data preparation mode
    """
    source_data_dirs: list[str]
    output_data_dir: str
    trg_person_id: str
    with_folds: bool