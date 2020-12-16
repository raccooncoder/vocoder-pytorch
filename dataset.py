from IPython import display
from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

from melspec import MelSpectrogramConfig, MelSpectrogram
from augment import RandomAudioCrop

class LJSpeech(Dataset):
    def __init__(self, X, config=MelSpectrogramConfig(), train=True):
        super().__init__()
        self.names = X
        self.train = train
        self.config = config
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        crop_size = 20000
        
        wav, sr = torchaudio.load('LJSpeech-1.1/wavs/' + self.names[idx] + '.wav')
        
        if self.train:
            wav = RandomAudioCrop(crop_size)(wav)
        else:
            wav = wav[:, :40000]
        
        if self.train:
            wav_proc = nn.Sequential(MelSpectrogram(self.config),
                                        #torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
                                        #torchaudio.transforms.TimeMasking(time_mask_param=100)
                                    )
        else:
            wav_proc = nn.Sequential(MelSpectrogram(self.config),
                                    )
        
        mel_spectrogram = wav_proc(wav)
         
        return mel_spectrogram.reshape(self.config.n_mels, -1), wav