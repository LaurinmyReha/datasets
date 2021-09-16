# needs to be rewriten so batch processing is supported properly
from audiofile import *
from textfile import *
from utils import *
import torchaudio

class Dataset:

    def __init__(self, name):
        self.name = name
        self.paths = get_paths_dataset(name)
        self.length = len(self.paths)
        self.transcripts = get_transcripts_dataset(name)
        self.chars_to_remove = None
        self.chars_to_replace = None
        self.replacement = None
        self.lowercase = True
        self.target_samplerate = 16000
        self.normalize_audio = True

    def get_element(self, index):
        waveform, samplerate = torchaudio.load(self.paths[index])
        waveform = waveform.numpy().flatten()
        return Audiofile(waveform, samplerate)

    def extract_all_chars(self):
        transcripts = list(self.transcripts.values())
        all_text = " ".join(transcripts)
        vocab = list(set(all_text))
        return vocab

    def preprocess_batch(self, indexes):
        batch = {}
        for index in indexes:
            audio_sample = self.get_element(index)
            filename = get_filename_from_file_path(self.paths[index])
            text_sample = self.transcripts[filename]
            text_sample = Textfile(text_sample)

            audio_preprocessed = audio_sample.preprocess(target_samplerate=self.target_samplerate,
                                                         normalize_audio=self.normalize_audio)
            text_preprocessed = text_sample.preprocess(self.chars_to_remove, self.chars_to_replace, self.replacement,
                                                       self.lowercase)
            filename = get_filename_from_file_path(self.paths[index])
            self.transcripts[filename] = text_preprocessed.text
            batch[index] = [audio_preprocessed, text_preprocessed]
        return batch

    def create_vocabulary(self):
        vocab = self.extract_all_chars()
        vocab_dict = {v: k for k, v in enumerate(vocab)}
        return vocab_dict











