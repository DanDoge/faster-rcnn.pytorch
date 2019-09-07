# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
#import model.utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.utils.config import cfg
import pdb
import PIL


class pascal3dimagenet(imdb):
  def __init__(self, version, image_set):
    imdb.__init__(self, 'pascal3dimagenet_' + version + '_' + image_set)
    self._image_set = image_set

    self._data_path = os.path.join(cfg.DATA_DIR, 'pascal3d+1.1')
    self._classes = ('__background__',  # always index 0
                     'aeroplane_imagenet', 'bicycle_imagenet', 'boat_imagenet',
                     'bottle_imagenet', 'bus_imagenet', 'car_imagenet', 'chair_imagenet',
                     'diningtable_imagenet',
                     'motorbike_imagenet',
                     'sofa_imagenet', 'train_imagenet', 'tvmonitor_imagenet')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.JPEG'
    self.objs = pickle.load(open("objs_imagenet_" + image_set + ".pkl", "rb"))
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': False,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    if self._image_set == "train":
        image_path = os.path.join(self._data_path, 'Images', self.objs[index][0]['label'],
                                  index + self._image_ext)
    elif self._image_set == "val":
        image_path = os.path.join(self._data_path, 'Images', self.objs[index][0]['name'],
                                  index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def image_id_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return i

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    image_index = list(tuple(self.objs))
    return image_index

  def append_flipped_images(self):
    #return
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all(), '{}, {}'.format(boxes[:, 0], boxes[:, 2])
      entry = {'boxes': boxes,
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True,
               'gt_viewpoints': 25 - self.roidb[i]['gt_viewpoints'],
               'gt_elevation': self.roidb[i]['gt_elevation']}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2
    return

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'pascal3d+1.1')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_pascal_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    objs = self.objs[index]
    num_objs = len(objs)

    imagesize = PIL.Image.open(self.image_path_from_index(index)).size
    width = imagesize[0]
    height = imagesize[1]

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    gt_viewpoints = np.zeros((num_objs), dtype=np.int32)
    gt_elevation = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      # Make pixel indexes 0-based
      x1 = obj['bbox'][0]
      y1 = obj['bbox'][1]
      x2 = obj['bbox'][2]
      y2 = obj['bbox'][3]
      if x1 > x2:
          t = x1
          x1 = x2
          x2 = t
      if y1 > y2:
          t = y1
          y1 = y2
          y2 = t
      if x1 <= 0:
          x1 = 1
      if y1 <= 0:
          y1 = 1
      if x2 >= width:
          x2 = width - 1
      if y2 >= height:
          y2 = height - 1
      if self._image_set == "train":
          cls = self._class_to_ind[obj['label'].lower().strip()]
      elif self._image_set == "val":
          cls = self._class_to_ind[obj['name'].lower().strip()]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      # similar to class labels, we assign 0 to bg rois
      gt_viewpoints[ix] = obj['viewpoint'] + 1
      gt_elevation[ix] = obj['elevation'] + 1
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
            'gt_viewpoints':gt_viewpoints,
            'gt_elevation': gt_elevation}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    # should be ./data/pascal3d/results/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    filedir = os.path.join(
      self._data_path,
      'results',
      'Main')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:d} {:d}\n'.
                    format(index, dets[k, -3],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1,
                           int(dets[k, -2]), int(dets[k, -1])))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._data_path,
      'Annotations',
      '{:s}.xml')
    '''
    imagesetfile = os.path.join(
      self._data_path,
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    '''
    imagesetfile = os.path.join(
        self._data_path,
        'Image_sets',
        '{:s}_' + self._image_set + '.txt'
    )
    cachedir = os.path.join(self._data_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True # if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      imageset_file = imagesetfile.format(cls)
      rec, prec, ap = voc_eval(
        filename, annopath, imageset_file, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric)
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()