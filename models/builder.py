from torch import nn

from utils import build_from_cfg
from .registry import (RGB_BACKBONES, DEPTH_BACKBONES, POINTCLOUD_BACKBONES, NECKS, HEADS, DETECTORS)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_rgb_backbone(cfg):
    return build(cfg, RGB_BACKBONES)


def build_depth_backbone(cfg):
    return build(cfg, DEPTH_BACKBONES)


def build_pointcloud_backbone(cfg):
    return build(cfg, POINTCLOUD_BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_detector(cfg):
    return build(cfg, DETECTORS)