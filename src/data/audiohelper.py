import librosa as lbr
import numpy as np

class AudioHelper:
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
    
    def augmentation(self, y:np.ndarray) -> np.ndarray:
        y = y.copy()
        # noise
        noise = np.random.normal(0, 0.002, len(y))
        y += noise

        # gain
        gain = np.random.uniform(0.9, 1.1)
        y *= gain

        # shift
        #shift = np.random.randint(
        #    -int(0.03 * self.SR),
        #    int(0.03 * self.SR)
        #)
        #y = np.roll(y, shift)

        return y

    def feature_extraction(self, audio_path: str, is_augmentation: bool):
        """
        Речь это комбинация частот
        Нужно выделить те частоты, которые человек слышит и различает говорящих
        """
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

        if is_augmentation:
            y = self.augmentation(y)

        # pre-emphasis
        # гласные это низкие частоты с сильной энергией
        # согласные это высокие частоты но быстро меняются, именно они создают общий ланшафд речи
        y = np.append(y[0], y[1:] - self.PRE_EMPHASIS * y[:-1])

        # MFCC: shape [n_mfcc, T]
        mfcc = self.mfcc(y)

        # delta (derivation throw time) and delta-delta
        delta = lbr.feature.delta(mfcc)
        delta2 = lbr.feature.delta(mfcc, order=2)

        # combine: [n_mfcc + delta + delta2, T]
        features = np.vstack([mfcc, delta, delta2])
        #features = mfcc
        features = features.T

        # normalization throw time
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        features = (features - mean) / std

        return features
