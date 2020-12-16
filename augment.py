import torch
from torch import nn
import random

class RandomAudioCrop(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        
        self.crop_size = crop_size
        
    def forward(self, wav):
        idx = random.randint(0, wav.shape[-1] - self.crop_size - 1)
        
        return wav[:, idx:idx+self.crop_size]