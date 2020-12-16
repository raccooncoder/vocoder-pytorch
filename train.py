def train(epoch, log_freq=50):
    model.train() #don't forget to switch between train and eval!
    
    running_loss = 0.0 #more accurate representation of current loss than loss.item()
    running_correct = 0.0
    
    for i, (mels, wavs) in enumerate(tqdm(train_loader)):
        mels, wavs = mels.to(device), wavs.to(device)
        targets =  MuLawEncoding()(wavs.squeeze())
        inp_wavs = torch.cat([torch.zeros(wavs.shape[0], 1, 1).to(device), wavs[:, :, :-1]], dim=2)
        inp_wavs = MuLawEncoding()(inp_wavs).float()

        optimizer.zero_grad()

        outputs = model(inp_wavs, mels)
        
        loss = error(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        pred = outputs.data.max(1, keepdim=True)[1]
        running_correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

        if (i + 1) % log_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                    epoch, (i+ 1) * mels.shape[0], len(train_loader.dataset),
                    100. * (i + 1) / len(train_loader), running_loss / log_freq, 
                    running_correct / log_freq / wavs.shape[0] / wavs.shape[-1]))
            
            wandb.log({"Loss": running_loss / log_freq,
                       "Accuracy": running_correct / log_freq / wavs.shape[0] / wavs.shape[-1]})
            
            running_loss = 0.0
            running_correct = 0.0
            
def test(epoch):
    model.eval()
    
    mel, wav = test_dataset[0]
    
    mel, wav = mel.to(device), wav.to(device)
    
    with torch.no_grad():
        pred = model.inference(mel.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    
    wandb.log({"examples": [wandb.Audio(pred, caption="Epoch {}".format(epoch), sample_rate=22050)]})
    #ipd.Audio(pred, rate=22050)
    
    return pred