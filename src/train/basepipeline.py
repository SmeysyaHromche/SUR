import joblib
class BasePipeline:

    def __call__(self):
        pass

    def store_log(self, path, txt:str) -> None:
        with open(path, "w") as f:
            f.write(txt)