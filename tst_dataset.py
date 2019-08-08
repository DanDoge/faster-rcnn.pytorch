# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
#from model.utils.net_utils import weights_normal_init, save_net, load_net, \
#      adjust_learning_rate, save_checkpoint, clip_gradient

#from model.faster_rcnn.vgg16 import vgg16
#from model.faster_rcnn.resnet import resnet


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

def tst_gt_box():
    imdb, roidb, ratio_list, ratio_index = combined_roidb("pascal3d_1.0_train")

    train_size = len(roidb)

    sampler_batch = sampler(train_size, 1)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                              sampler=sampler_batch, num_workers=0)

    gt_boxes = torch.FloatTensor(1)
    gt_boxes = Variable(gt_boxes)
    for data in dataloader:
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        print(gt_boxes)
        #break;


import pickle

def tst():
    cfg.EXP_DIR = 'vgg16'
    save_name = 'faster_rcnn_10'
    imdb, roidb, ratio_list, ratio_index = combined_roidb("pascal3d_1.0_val", False)
    output_dir = get_output_dir(imdb, save_name)
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'rb') as f:
        all_boxes = pickle.load(f)
    imdb.evaluate_detections(all_boxes, output_dir)

if __name__ == '__main__':
    tst_gt_box()
