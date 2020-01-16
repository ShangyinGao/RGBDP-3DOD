from mmcv import Config
import time
import argparse
import torch
import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import torch.nn as nn
from torch.autograd import Variable

from datasets import build_dataloader
from datasets.roi_data_layer.config import cfg as cfg_roi
from models import build_detector
from utils.logger import get_root_logger
from models.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
import pdb


def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description='Train RGBD-P 3D detector')
	parser.add_argument('config', help='train config file path')

	parser.add_argument('--save_dir', dest='save_dir',
						help='directory to save models', default="output",
						type=str)

	args = parser.parse_args()
	return args


def main():
	args = parse_args()
	cfg = Config.fromfile(args.config)

	# init logger before other steps
	logger = get_root_logger(cfg.log_level)
	logger.info('Config: {}'.format(cfg.text))

	dataloader=build_dataloader(cfg)

	model = build_detector(cfg.model)


	# initilize the tensor holder here.
	im_left_data = Variable(torch.FloatTensor(1).cuda())
	im_right_data = Variable(torch.FloatTensor(1).cuda())
	im_info = Variable(torch.FloatTensor(1).cuda())
	num_boxes = Variable(torch.LongTensor(1).cuda())
	point_cloud = Variable(torch.FloatTensor(1).cuda())
	gt_boxes_left = Variable(torch.FloatTensor(1).cuda())
	gt_boxes_right = Variable(torch.FloatTensor(1).cuda())
	gt_boxes_merge = Variable(torch.FloatTensor(1).cuda())
	gt_dim_orien = Variable(torch.FloatTensor(1).cuda())

	uncert = Variable(torch.rand(6).cuda(), requires_grad=True)
	torch.nn.init.constant(uncert, -1.0)
	lr = cfg_roi.TRAIN.LEARNING_RATE

	params = []
	for key, value in dict(model.named_parameters()).items():
		if value.requires_grad:
			if 'bias' in key:
				params += [{'params':[value],'lr':lr*(cfg_roi.TRAIN.DOUBLE_BIAS + 1), \
						 'weight_decay': cfg_roi.TRAIN.BIAS_DECAY and cfg_roi.TRAIN.WEIGHT_DECAY or 0}]
			else:
				params += [{'params':[value],'lr':lr, 'weight_decay': cfg_roi.TRAIN.WEIGHT_DECAY}]
	params += [{'params':[uncert], 'lr':lr}]

	optimizer = torch.optim.SGD(params, momentum=cfg_roi.TRAIN.MOMENTUM)

	model.cuda()

	train_size = len(dataloader)
	iters_per_epoch = int(train_size / cfg.batch_size)
	total_step = 0
	for epoch in range(0, cfg.total_epochs + 1):
	
		model.train()

		# if epoch % (args.lr_decay_step + 1) == 0:
		if epoch in cfg.lr_config.step:
			adjust_learning_rate(optimizer, 0.1)
			lr *= 0.1

		data_iter = iter(dataloader)
		for step in range(iters_per_epoch):
			data = next(data_iter)
			im_left_data.data.resize_(data[0].size()).copy_(data[0])
			im_right_data.data.resize_(data[1].size()).copy_(data[1])
			im_info.data.resize_(data[2].size()).copy_(data[2])
			gt_boxes_left.data.resize_(data[3].size()).copy_(data[3])
			gt_boxes_right.data.resize_(data[4].size()).copy_(data[4])
			gt_boxes_merge.data.resize_(data[5].size()).copy_(data[5])
			gt_dim_orien.data.resize_(data[6].size()).copy_(data[6])
			num_boxes.data.resize_(data[8].size()).copy_(data[8]) 
			point_cloud.data.resize_(data[9].size()).copy_(data[9])

			start = time.time()
			model.zero_grad()
			rois_left, rois_right, cls_prob, bbox_pred, dim_orien_pred, \
			rpn_loss_cls, rpn_loss_box_left_right,\
			RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, rois_label =\
			model(im_left_data, im_right_data, im_info, gt_boxes_left, gt_boxes_right, \
					   gt_boxes_merge, gt_dim_orien, num_boxes, point_cloud)


 
			loss = rpn_loss_cls.mean() * torch.exp(-uncert[0]) + uncert[0] +\
					rpn_loss_box_left_right.mean() * torch.exp(-uncert[1]) + uncert[1] +\
					RCNN_loss_cls.mean() * torch.exp(-uncert[2]) + uncert[2]+\
					RCNN_loss_bbox.mean() * torch.exp(-uncert[3]) + uncert[3] +\
					RCNN_loss_dim_orien.mean() * torch.exp(-uncert[4]) + uncert[4] #+\
			uncert_data = uncert.data
			# log_string('uncert: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' \
			# 		  %(uncert_data[0], uncert_data[1], uncert_data[2], uncert_data[3], uncert_data[4], uncert_data[5])) 

			optimizer.zero_grad()
			loss.backward()
			# clip_gradient(model, 10.)
			optimizer.step()

			loss_rpn_cls = rpn_loss_cls.item()
			loss_rpn_box_left_right = rpn_loss_box_left_right.item()
			loss_rcnn_cls = RCNN_loss_cls.item()
			loss_rcnn_box = RCNN_loss_bbox.item()
			loss_rcnn_dim_orien = RCNN_loss_dim_orien.item()
			fg_cnt = torch.sum(rois_label.data.ne(0))
			bg_cnt = rois_label.data.numel() - fg_cnt

			# log_string('[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e'\
			# 	  %(epoch, step, iters_per_epoch, loss.item(), lr))
			# log_string('\t\t\tfg/bg=(%d/%d), time cost: %f' %(fg_cnt, bg_cnt, end-start))
			# log_string('\t\t\trpn_cls: %.4f, rpn_box_left_right: %.4f, rcnn_cls: %.4f, rcnn_box_left_right %.4f,dim_orien %.4f' \
			# 	  %(loss_rpn_cls, loss_rpn_box_left_right, loss_rcnn_cls, loss_rcnn_box, loss_rcnn_dim_orien))

			del loss, rpn_loss_cls, rpn_loss_box_left_right, RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien#, RCNN_loss_kpts

		# save_name = os.path.join(output_dir, 'checkpoint_{}_{}.pth'.format(epoch, step))
		# save_checkpoint({
		# 	'epoch': epoch + 1,
		# 	'model': model.state_dict(),
		# 	'optimizer': optimizer.state_dict(),
		# 	'uncert':uncert.data,
		# }, save_name)

		# log_string('save model: {}'.format(save_name)) 
		# end = time.time()
		# log_string('time %.4f' %(end - start))

if __name__ == "__main__":
	main()