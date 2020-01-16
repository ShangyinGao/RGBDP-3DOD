import torch
import torch.nn as nn
from ..registry import RGB_BACKBONES


@RGB_BACKBONES.register_module
class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        pass

    def forward(self):
        pass