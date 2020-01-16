from utils import Registry

# BACKBONES = Registry('backbone')
# NECKS = Registry('neck')
# ROI_EXTRACTORS = Registry('roi_extractor')
# SHARED_HEADS = Registry('shared_head')
# HEADS = Registry('head')
# LOSSES = Registry('loss')
# DETECTORS = Registry('detector')

RGB_BACKBONES = Registry('rgb_backbone')
DEPTH_BACKBONES = Registry('depth_backbone')
POINTCLOUD_BACKBONES = Registry('pointcloud_backbone')
NECKS = Registry('neck')
HEADS = Registry('head') # 3d detector in report
DETECTORS = Registry('detector') # whole network