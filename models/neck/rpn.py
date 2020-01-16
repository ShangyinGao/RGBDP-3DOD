import torch
import torch.nn as nn
from ..registry import NECKS


@NECKS.register_module
class rpn(nn.Module):
    def __init__(self):
        super(rpn, self).__init__()
        pass

    def forward(self):
        pass