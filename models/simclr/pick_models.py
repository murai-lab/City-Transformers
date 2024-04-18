import torch.nn as nn
import torchvision.models as models
import torch
from collections import OrderedDict
from torch.nn import functional as F
import os


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers):
        super().__init__(block, layers)
        print('** Using avgpool **')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
class ResNet50Base(ResNetEncoder):
    def __init__(self):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3])

def fix_model(model, fixed):
    if fixed:
        for param in model.parameters():
            param.requires_grad = False

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class ResNet50P(nn.Module):
    def __init__(self, encoder_dim, proj_dim):
        super().__init__()
        self.convnet = ResNet50Base()
        self.encoder_dim = encoder_dim
        self.proj_dim = proj_dim
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]
        self.projection = nn.Sequential(OrderedDict(projection_layers))

    def forward(self, img):
        h = self.convnet(img)
        return self.projection(h)
    
    
class ResNet50(nn.Module):
    def __init__(self, numofcat, weighted=False, path=None, fixed=True, device='cpu',encoder_dim=2048, proj_dim=128):
        super().__init__()
        self.path = path
        self.fixed = fixed

        self.weighted = weighted
        if self.weighted:
            total = 19712+4881+2915+1252
            self.weight = torch.Tensor([total/19712, total/4881, total/2915, total/1252]).to(device)

        # load weight
        self.model  = ResNet50P(encoder_dim, proj_dim)
        if self.path is None:
            pass
        elif os.path.exists(self.path):
            ckpt = torch.load(self.path, map_location=device)
            self.model.load_state_dict(ckpt['state_dict'])
        else:
            raise NotImplementedError

        fix_model(self.model, fixed)

        self.classifier = nn.Linear(proj_dim, numofcat)
        self.classifier.weight.data.zero_()
        self.classifier.bias.data.zero_()

    def forward(self, img, y):
        h = self.model(img)
        p = self.classifier(h)
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
    model = ResNet50(2, 'yourpath.pth.tar')
    print(model)

    # print(model)
    a = torch.rand((2, 3, 224, 224))
    b = torch.randint(0, 1, (2,))
    print(a.size())
    print(b)
    print(model(a, b))
    
