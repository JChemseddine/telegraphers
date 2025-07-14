import torch
import torch.nn as nn


class TorchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,x,t):
       
        #x.requires_grad_(True)
        
        time = t.repeat(x.shape[0])[:, None].float()
       
        out = self.model(x.reshape(x.shape[0],3,32,32),time.squeeze(1)).reshape(x.shape[0],-1)#torch.cat([x,time],1))
            
        return out

