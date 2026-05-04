# author: Myron Kukhta (xkukht01)

import librosa as lbr
import numpy as np

class AudioHelper:
    """
    A helper class for working with audio data
    """
    SR = 16_000
    SPEACH_START_PEAK_ENERGY = 20
    PRE_EMPHASIS = 0.95
    N_MFCC = 16

    def __init__(self):
        pass
    
    
    def mfcc(self, y:np.ndarray) -> np.ndarray:
        return lbr.feature.mfcc(
            y=y,
            sr=self.SR,
            n_mfcc=self.N_MFCC,
            n_fft=int(0.025 * self.SR),      # 25 ms window
            hop_length=int(0.010 * self.SR), # 10 ms step
            window="hamming"
        )

    def feature_extraction(self, audio_path: str, is_augmentation: bool) -> None:
        y, _ = lbr.load(  # [time_in_sec * SR] amplituda score
            audio_path, 
            sr=self.SR, 
            mono=True
        )
        # remove silence
        y, _ = lbr.effects.trim(
            y, 
            top_db=self.SPEACH_START_PEAK_ENERGY
        )

        # pre-emphasis
        y = np.append(y[0], y[1:] - self.PRE_EMPHASIS * y[:-1])

        # MFCC: shape [n_mfcc, T]
        mfcc = self.mfcc(y)

        # delta (derivation throw time)
        delta = lbr.feature.delta(mfcc)

        # combine: [n_mfcc + delta, T]
        features = np.vstack([mfcc, delta])
        features = features.T

        return features
