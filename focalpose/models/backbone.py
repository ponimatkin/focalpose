import torch
from torch import nn
import torchvision.models


class Backbone(nn.Module):
    def __init__(self, model, pretrained=True):
        super(Backbone, self).__init__()
        if 'res' in model:
            if hasattr(torchvision.models, model):
                backbone = getattr(torchvision.models, model)(pretrained=pretrained)
            else:
                raise ValueError('Unknown backbone')
        else:
            raise ValueError('Only ResNe(x)t models are supported')

        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        conv1_weights_6d = self.backbone[0].weight.repeat(1, 2, 1, 1)
        self.backbone[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.backbone[0].weight = torch.nn.Parameter(conv1_weights_6d)

        if isinstance(self.backbone[-1][-1], torchvision.models.resnet.BasicBlock):
            self.n_features = self.backbone[-1][-1].conv2.out_channels
        elif isinstance(self.backbone[-1][-1], torchvision.models.resnet.Bottleneck):
            self.n_features = self.backbone[-1][-1].conv3.out_channels

    def forward(self, x):
        return self.backbone(x)