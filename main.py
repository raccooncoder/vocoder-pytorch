import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from sklearn.metrics import average_precision_score
from torchaudio.transforms import MuLawEncoding, MuLawDecoding

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

from dataset import LJSpeech
from melspec import MelSpectrogramConfig

num_epochs = 20
batch_size = 16

import wandb
wandb.init(project="tts-dlaudio")

# reproducibility
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_seed(13)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device='cpu'

print(device)

df = pd.read_csv("LJSpeech-1.1/metadata.csv", names=['id', 'gt', 'gt_letters_only'], sep="|")
df = df.dropna()

X_train, X_test  = train_test_split(list(df['id']), train_size=.9)

train_dataset = LJSpeech(X_train, train=True)
test_dataset = LJSpeech(X_test, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

model = WaveNet(MelSpectrogramConfig())

model = model.to(device)

learning_rate = 0.001

error = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(num_epochs): 
    train(epoch)
    if (epoch + 1) % 4 == 0:
        test(epoch)