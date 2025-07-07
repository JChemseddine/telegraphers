import torch
import torch.nn as nn


class TorchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,x,t):
       
        #x.requires_grad_(True)
        
        time = t.repeat(x.shape[0])[:, None].float()
       
        out = self.model(x.reshape(x.shape[0],1,28,28),time.squeeze(1)).reshape(x.shape[0],-1)#torch.cat([x,time],1))
            
        return out




class ODEWrapper(torch.nn.Module):
    def __init__(self, fmap):
        super().__init__()
        self.fmap = fmap

    def forward(self, t, x):
        return self.fmap(x, t)
