# author: Myron Kukhta (xkukht01)

import numpy as np

from .productionconfig import ProductionConfig
from src.model import Model

class BaseProductionPipeline:
    """
    A base class for production pipeline of binary classifier.
    """
    def __init__(self, config:ProductionConfig):
        self.config = config
    
    def store_classification_output(self, file_path, file_ids, scores, predictions):
        with open(file_path, "w", encoding="ascii") as f:
            for file_id, score, prediction in zip(file_ids, scores, predictions):
                f.write(f"{file_id} {float(score)} {int(prediction)}\n")
            print(f'Log of classification successfuly saved on path: {file_path}\n')

    def get_model(self) -> Model:
        raise NotImplementedError("Implement me")
    
    def get_data_for_evaluation(self):
        raise NotImplementedError("Implement me")
    
    def run(self):
        print("Preparation of model\n")
        model = self.get_model()
        
        print("Data reading\n")
        X, file_ids = self.get_data_for_evaluation()
        
        print("Data scoring\n")
        scores = model.score(X, file_ids)
        print("Data prediction\n")
        predictions = model.predict(X, file_ids)
        
        file_ids = np.array(list(dict.fromkeys(file_ids)))
        
        self.store_classification_output(
            self.config.classification_out_path, 
            file_ids, 
            scores, 
            predictions
        )