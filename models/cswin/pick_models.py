import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.insert(0, 'yourpath)
from models.cswin import CSWin_64_12211_tiny_224 as CSWin_T



def fix_model(model, fixed):
    if fixed:
        for param in model.parameters():
            param.requires_grad = False


class CSWin_tiny(nn.Module):
    def __init__(self, numofcat, weighted=False, path=None, fixed=True, device='cuda'):
        super().__init__()
        self.path = path
        self.fixed = fixed

        self.weighted = weighted
        if self.weighted:
            total = 19712+4881+2915+1252
            self.weight = torch.Tensor([total/19712, total/4881, total/2915, total/1252]).to(device)

        # load weight
        self.model  = CSWin_T()
        if self.path is None:
            pass
        elif os.path.exists(self.path):
            ckpt = torch.load(self.path, map_location=device)
            print(ckpt.keys())
            self.model.load_state_dict(ckpt['state_dict_ema'])
        else:
            raise NotImplementedError

        fix_model(self.model, fixed)

        self.model.head = nn.Linear(512, numofcat)
        self.model.head.weight.data.zero_()
        self.model.head.bias.data.zero_()

    def forward(self, img, y):
        p = self.model(img)
        if self.weighted:
            loss = F.cross_entropy(p, y, weight=self.weight)
        else:
            loss = F.cross_entropy(p, y)
        # print(p.argmax(1))
        return {
            'loss': loss,
            'predictions': p,
        }


if __name__ == '__main__':
    model = CSWin_tiny(4, path='/home/yzhang37/CSWin-Transformer/models/cswin_tiny_224.pth')
    print(model)
    model.to('cuda')
    # print(model)
    a = torch.rand((2, 3, 224, 224)).to('cuda')
    b = torch.randint(0, 1, (2,)).to('cuda')
    print(a.size())
    print(b)
    print(model(a, b))
    
