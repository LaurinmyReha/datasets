import librosa
import numpy as np


class Audiofile:

    def __init__(self, waveform, samplerate):
        self.samplerate = samplerate
        self.waveform = waveform

    def downsample(self, target_samplerate=16000):
        waveform = librosa.resample(self.waveform, self.samplerate, target_samplerate, res_type='kaiser_best', fix=True)
        return Audiofile(waveform, target_samplerate)

    def z_scale_waveform(self):
        zero_mean_waveform = self.waveform - np.mean(self.waveform)
        z_scale_waveform = zero_mean_waveform / np.std(self.waveform)
        return Audiofile(z_scale_waveform, self.samplerate)

    def preprocess(self, target_samplerate=16000, normalize_audio=True):
        preprocessed_waveform = self.downsample(target_samplerate=target_samplerate)
        if (normalize_audio):
            preprocessed_waveform = preprocessed_waveform.z_scale_waveform()
        return preprocessed_waveform


