import torch
from torch import nn


class ControlledSFCNN(nn.Module):
    """Same parameter allocation in every arm; metadata differs by arm.

    Returns the FULL predicted normalized channel, including the residual skip.
    """
    def __init__(self, mode, width=64):
        super().__init__()
        self.mode=mode
        self.convs=nn.ModuleList([nn.Conv2d(32 if i==0 else width,width,3,padding=1) for i in range(5)])
        self.norms=nn.ModuleList([nn.BatchNorm2d(width) for _ in range(5)])
        self.encoder=nn.Sequential(nn.Linear(1200,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.film=nn.ModuleList([nn.Linear(32,2*width) for _ in range(5)])
        self.out=nn.Conv2d(width,4,1)
        for f in self.film: nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)

    def forward(self,x,g,alpha):
        if self.mode=='geometry': metadata=g
        elif self.mode=='ratio': metadata=alpha[:,None].expand(-1,1200)
        else: metadata=torch.zeros_like(g)
        embedding=self.encoder(metadata)
        h=x
        for conv,norm,film in zip(self.convs,self.norms,self.film):
            gain,bias=film(embedding).chunk(2,dim=1)
            h=torch.relu(norm(conv(h))*(1+gain[:,:,None,None])+bias[:,:,None,None])
        return x[:,:4]+self.out(h)


def nmse(pred,target):
    return (pred-target).square().sum((1,2,3))/target.square().sum((1,2,3)).clamp_min(1e-12)
