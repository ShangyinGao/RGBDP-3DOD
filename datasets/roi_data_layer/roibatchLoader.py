
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch
import cv2

from datasets.roi_data_layer.config import cfg
from models.backbone_pointcloud.p_config import cfg as cfg_p
from datasets.roi_data_layer.minibatch import get_minibatch, get_minibatch
from models.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes)

    data_left = torch.from_numpy(blobs['data_left'])
    data_right = torch.from_numpy(blobs['data_right'])
    pts_rect = np.fromfile(self._roidb[index_ratio]['point_cloud'], dtype=np.float32).reshape(-1, 4)
    pts_intensity = pts_rect[:, 3]
    im_info = torch.from_numpy(blobs['im_info'])

    if self.training:
        boxes_all = np.concatenate((blobs['gt_boxes_left'], blobs['gt_boxes_right'],blobs['gt_boxes_merge'],\
                                    blobs['gt_dim_orien'], blobs['gt_kpts']), axis=1)
        
        np.random.shuffle(boxes_all) 
        
        gt_boxes_left = torch.from_numpy(boxes_all[:,0:5])
        gt_boxes_right = torch.from_numpy(boxes_all[:,5:10])
        gt_boxes_merge = torch.from_numpy(boxes_all[:,10:15])
        gt_dim_orien = torch.from_numpy(boxes_all[:,15:20])
        gt_kpts = torch.from_numpy(boxes_all[:,20:26])

        num_boxes = min(gt_boxes_left.size(0), self.max_num_box) 

        data_left = data_left[0].permute(2, 0, 1).contiguous()
        data_right = data_right[0].permute(2, 0, 1).contiguous()
        im_info = im_info.view(3) 

        gt_boxes_left_padding = torch.FloatTensor(self.max_num_box, gt_boxes_left.size(1)).zero_()
        gt_boxes_left_padding[:num_boxes,:] = gt_boxes_left[:num_boxes]
        gt_boxes_left_padding = gt_boxes_left_padding.contiguous()

        gt_boxes_right_padding = torch.FloatTensor(self.max_num_box, gt_boxes_right.size(1)).zero_()
        gt_boxes_right_padding[:num_boxes,:] = gt_boxes_right[:num_boxes]
        gt_boxes_right_padding = gt_boxes_right_padding.contiguous() 

        gt_boxes_merge_padding = torch.FloatTensor(self.max_num_box, gt_boxes_merge.size(1)).zero_()
        gt_boxes_merge_padding[:num_boxes,:] = gt_boxes_merge[:num_boxes]
        gt_boxes_merge_padding = gt_boxes_merge_padding.contiguous()

        gt_dim_orien_padding = torch.FloatTensor(self.max_num_box, gt_dim_orien.size(1)).zero_()
        gt_dim_orien_padding[:num_boxes,:] = gt_dim_orien[:num_boxes]
        gt_dim_orien_padding = gt_dim_orien_padding.contiguous()

        gt_kpts_padding = torch.FloatTensor(self.max_num_box, gt_kpts.size(1)).zero_()
        gt_kpts_padding[:num_boxes,:] = gt_kpts[:num_boxes]
        gt_kpts_padding = gt_kpts_padding.contiguous()

        # point cloud
        # generate inputs
        if cfg_p.npoints < len(pts_rect):
            pts_depth = pts_rect[:, 2]
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, cfg_p.npoints - len(far_idxs_choice), replace=False)

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(pts_rect), dtype=np.int32)
            if cfg_p.npoints > len(pts_rect):
                extra_choice = np.random.choice(choice, cfg_p.npoints - len(pts_rect), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        ret_pts_rect = pts_rect[choice, :]
        ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        pts_input = ret_pts_rect[:, :3]
        
        ######

        return data_left, data_right, im_info, gt_boxes_left_padding, gt_boxes_right_padding,\
               gt_boxes_merge_padding, gt_dim_orien_padding, gt_kpts_padding, num_boxes, pts_input
    else: 
        data_left = data_left[0].permute(2, 0, 1).contiguous()
        data_right = data_right[0].permute(2, 0, 1).contiguous()
        im_info = im_info.view(3) 


        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data_left, data_right, im_info, gt_boxes, gt_boxes, gt_boxes, gt_boxes, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
