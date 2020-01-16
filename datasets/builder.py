# import copy

import torch
from torch.utils.data.sampler import Sampler

import pdb

# from utils import build_from_cfg
# from .dataset_wrappers import ConcatDataset, RepeatDataset
# from .registry import DATASETS


# def _concat_dataset(cfg, default_args=None):
#     ann_files = cfg['ann_file']
#     img_prefixes = cfg.get('img_prefix', None)
#     seg_prefixes = cfg.get('seg_prefixes', None)
#     proposal_files = cfg.get('proposal_file', None)

#     datasets = []
#     num_dset = len(ann_files)
#     for i in range(num_dset):
#         data_cfg = copy.deepcopy(cfg)
#         data_cfg['ann_file'] = ann_files[i]
#         if isinstance(img_prefixes, (list, tuple)):
#             data_cfg['img_prefix'] = img_prefixes[i]
#         if isinstance(seg_prefixes, (list, tuple)):
#             data_cfg['seg_prefix'] = seg_prefixes[i]
#         if isinstance(proposal_files, (list, tuple)):
#             data_cfg['proposal_file'] = proposal_files[i]
#         datasets.append(build_dataset(data_cfg, default_args))

#     return ConcatDataset(datasets)


# def build_dataset(cfg, default_args=None):
#     if isinstance(cfg, (list, tuple)):
#         dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
#     elif cfg['type'] == 'RepeatDataset':
#         dataset = RepeatDataset(
#             build_dataset(cfg['dataset'], default_args), cfg['times'])
#     elif isinstance(cfg['ann_file'], (list, tuple)):
#         dataset = _concat_dataset(cfg, default_args)
#     else:
#         dataset = build_from_cfg(cfg, DATASETS, default_args)

#     return dataset


# old shool way to load kIITT
class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


def build_dataloader(cfg):
    """
    TODO: this is not the optimal way to load dataset
    try to switch to `build_dataset`
    """
    from .roi_data_layer.roidb import combined_roidb
    from .roi_data_layer.roibatchLoader import roibatchLoader

    batch_size = cfg.batch_size

    imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_train')
    train_size = len(roidb)
    sampler_batch = sampler(train_size, batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                           imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler_batch, num_workers=8)

    return dataloader