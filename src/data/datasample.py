class DataSample:
    """ 
    Wrap for combinations of different data types (image, audio, label)
    """
    def __init__(self, image, audio, label):
        self.image = image
        self.audio = audio
        self.label = label