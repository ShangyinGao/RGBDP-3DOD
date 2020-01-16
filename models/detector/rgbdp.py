import torch
import torch.nn as nn

from .. import builder
from ..registry import DETECTORS

import pdb


@DETECTORS.register_module
class rgbdp(nn.Module):
    def __init__(self, backbone_rgb, backbone_depth, backbone_pointcloud, neck, head):
        super(rgbdp, self).__init__()
        self.rgb = builder.build_rgb_backbone(backbone_rgb)
        self.pointcloud = builder.build_pointcloud_backbone(backbone_pointcloud)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)


    def forward(self, *input):
        point_cloud_feautre = self.pointcloud(input[-1])
        return self.head(*input[:-1], point_cloud_feautre)