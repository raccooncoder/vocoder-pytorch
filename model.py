import torch
from torch import nn 
import torch.nn.functional as F
from tqdm import tqdm

from melspec import MelSpectrogramConfig
from torchaudio.transforms import MuLawDecoding

class WaveNetBlock(nn.Module):
    def __init__(self,
                 dilation,
                 melspec_config=MelSpectrogramConfig(),
                 residual_channels=120,
                 skip_channels=240,
                 ):
        super().__init__()
        
        self.residual_channels = residual_channels
        
        self.dilation = dilation
        self.gated_act = lambda x1, y1, x2, y2: torch.tanh(x1 + y1) * torch.sigmoid(x2 + y2)
        
        self.cond_conv = nn.Conv1d(melspec_config.n_mels, 2 * residual_channels, kernel_size=1)
        self.dilated_conv = nn.Conv1d(residual_channels, 2  * residual_channels, kernel_size=2, 
                                      dilation=dilation, padding=dilation)
        
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        
    def forward(self, cond, wav):
        cond_out = self.cond_conv(cond)
        dilated_out = self.dilated_conv(wav)[:, :, :-self.dilation]
        
        cond_out_sigmoid = cond_out[:, :self.residual_channels, :]
        cond_out_tanh = cond_out[:, self.residual_channels:, :]
        dilated_out_sigmoid = dilated_out[:, :self.residual_channels, :]
        dilated_out_tanh = dilated_out[:, self.residual_channels:, :]
        
        out = self.gated_act(cond_out_sigmoid, 
                             dilated_out_sigmoid, 
                             cond_out_tanh, 
                             dilated_out_tanh)
        
        skip_out = self.skip_conv(out)
        res_out = self.res_conv(out) + wav
        
        return skip_out, res_out
        
class WaveNet(nn.Module):
    def __init__(self,
                 device, 
                 melspec_config,
                 upsample_kernel_size=800,
                 upsample_stride=256,
                 skip_channels=240, 
                 residual_channels=120,
                 num_dilations=8,
                 num_blocks=16,
                 num_classes=256
                 ):
        super().__init__()
        
        self.device = device
        self.melspec_config = melspec_config
        self.skip_channels = skip_channels
        
        self.upsample = nn.ConvTranspose1d(melspec_config.n_mels, 
                                          melspec_config.n_mels, 
                                          upsample_kernel_size, 
                                          upsample_stride, 
                                          padding = (upsample_kernel_size + 4 * upsample_stride - 
                                                     melspec_config.win_length) // 2,
                                          output_padding=32)
                                          
        
        self.wav_conv = nn.Conv1d(1, residual_channels, kernel_size=1)
        
        self.block_list = nn.ModuleList()
        
        self.receptive_field = 1
        
        for i in range(num_blocks):
            dilation = 2 ** (i % num_dilations)
            self.receptive_field += dilation 
            block = WaveNetBlock(dilation).to(device)
            self.block_list.append(block)
            
        self.out_conv = nn.Conv1d(skip_channels, num_classes, kernel_size=1)
        self.end_conv = nn.Conv1d(num_classes, num_classes, kernel_size=1)
        
            
    def forward(self, wav, mel=None, cond=None):
        if mel is not None:
            cond = self.upsample(mel)
            
        res_out = self.wav_conv(wav)
        
        skip_sum = torch.zeros(wav.shape[0], self.skip_channels, wav.shape[-1]).to(self.device)
        
        for block in self.block_list:
            skip_out, res_out = block(cond, res_out)
            skip_sum += skip_out
        
        out = self.out_conv(F.relu(skip_sum))
        end = self.end_conv(F.relu(out))
        
        return end
    
    def inference(self, mel):
        cond = self.upsample(mel)
        
        pred = torch.zeros(cond.shape[0], cond.shape[-1]).to(self.device)
        
        for i in tqdm(range(cond.shape[-1])):
            wav = pred[:, max(0, i + 1 - self.receptive_field) : i + 1]
            wav = wav[:, max(0, i + self.receptive_field - cond.shape[-1]) :]
            
            outputs = self.forward(wav.unsqueeze(1), cond=cond[:, :, i:i + wav.shape[-1]])
            
            dist = torch.distributions.Categorical(probs=F.softmax(outputs[:, :, -1].squeeze()))
            outputs = dist.sample()
            #print(outputs.shape)
            
            cur_pred = MuLawDecoding()(outputs)
            pred[:, i] = cur_pred
            
        return pred