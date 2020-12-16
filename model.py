class WaveNetBlock(nn.Module):
    def __init__(self,
                 dilation,
                 melspec_config=MelSpectrogramConfig(),
                 residual_channels=120,
                 skip_channels=240,
                 ):
        super().__init__()
        
        self.dilation = dilation
        self.gated_act = lambda x: torch.tanh(x) * torch.sigmoid(x)
        
        self.cond_conv = nn.Conv1d(melspec_config.n_mels, 2 * residual_channels, kernel_size=1)
        self.dilated_conv = nn.Conv1d(residual_channels, 2  * residual_channels, kernel_size=2, 
                                      dilation=dilation, padding=dilation)
        
        self.skip_conv = nn.Conv1d(2 * residual_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.Conv1d(2 * residual_channels, residual_channels, kernel_size=1)
        
    def forward(self, cond, wav):
        cond_out = self.cond_conv(cond)
        dilated_out = self.dilated_conv(wav)[:, :, :-self.dilation]
        
        out = self.gated_act(cond_out + dilated_out)
        
        skip_out = self.skip_conv(out)
        res_out = self.res_conv(out) + wav
        
        return skip_out, res_out
        
class WaveNet(nn.Module):
    def __init__(self, 
                 melspec_config,
                 upsample_kernel_size=800,
                 upsample_stride=256,
                 skip_channels=240, 
                 residual_channels=120,
                 audio_channels=256,
                 num_dilations=8,
                 num_blocks=16,
                 num_classes=256
                 ):
        super().__init__()
        
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
        
        self.block_list = []
        
        self.receptive_field = 1
        
        for i in range(num_blocks):
            dilation = 2 ** (i % 8)
            self.receptive_field += dilation 
            block = WaveNetBlock(dilation).to(device)
            self.block_list.append(block)
            
        self.out_conv = nn.Conv1d(skip_channels, num_classes, kernel_size=1)
        self.end_conv = nn.Conv1d(num_classes, num_classes, kernel_size=1)
        
            
    def forward(self, wav, mel=None, cond=None):
        if mel is not None:
            cond = self.upsample(mel)
            
        res_out = self.wav_conv(wav)
        
        skip_sum = torch.zeros(wav.shape[0], self.skip_channels, wav.shape[-1]).to(device)
        
        for block in self.block_list:
            skip_out, res_out = block(cond, res_out)
            skip_sum += skip_out
        
        out = self.out_conv(F.relu(skip_sum))
        end = self.end_conv(F.relu(out))
        
        return end
    
    def inference(self, mel):
        cond = self.upsample(mel)
        
        pred = torch.zeros(cond.shape[0], cond.shape[-1]).to(device)
        
        for i in tqdm(range(cond.shape[-1])):
            wav = pred[:, max(0, i + 1 - self.receptive_field) : i + 1]
            wav = wav[:, max(0, i + self.receptive_field - cond.shape[-1]) :]
            
            outputs = self.forward(wav.unsqueeze(1), cond=cond[:, :, i:i + wav.shape[-1]])
            
            cur_pred = MuLawDecoding()(outputs[:, :, -1].data.max(1, keepdim=True)[1])
            pred[:, i] = cur_pred
            
        return pred