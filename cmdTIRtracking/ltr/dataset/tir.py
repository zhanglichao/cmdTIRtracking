import os
import os.path
import numpy as np
import torch
import csv
import cv2
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader,opencv_loader
from ltr.admin.environment import env_settings


class Tir(BaseVideoDataset):

    def __init__(self, root=None, image_loader=opencv_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().tir_dir if root is None else root
        super().__init__('TIR', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the tir root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'tir_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'tir_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

    def get_name(self):
        return 'tir'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        # with open(os.path.join(self.root, 'list.txt')) as f:
            # dir_list = list(csv.reader(f))
        
        # dir_list = [dir_name[0] for dir_name in dir_list]
        img_list = []
        for roots, dirs, files in os.walk(self.root):
            for file in files:
                img_list.append(os.path.join(self.root,file))
        return img_list

    def _read_bb_anno(self, seq_path):
        # 1. central region
        gt = torch.Tensor([[206.0,540.0,100.0,200.0]])
        return gt
        
        # 2. randomly sampling
        # it can be seen in 4. detector (Yolo)

        # 3. detector (FairMot)
        # dir = seq_path[:35]
        # name = seq_path[41:] + '.txt'
        # bb_anno_file = os.path.join(dir, 'bbox1', name)
        # gt = open(bb_anno_file)
        # gt = gt.readlines()
        # n = 0
        # for p in gt:
            # xx = p.split()
            # for i in range(5):
                # xx[i] = float(xx[i])
            # gt[n] = xx
            # n = n+1
        # m=0
        # for s in gt: # T: x+640
            # w = s[2] - s[0]
            # h = s[3] - s[1]
            # if s[5] == 'tcar' or s[5] == 'tperson':
                # s[0] = s[0] + 640
            # s[2] = w
            # s[3] = h
            # gt[m] = s[:4]
            # m = m+1
        # return torch.tensor(gt)

        # 4. detector (Yolo)
        # dir = seq_path[:35]
        # name = seq_path[41:]
        # txt = name[:-3] + 'txt'
        # bb_anno_rgb = os.path.join(dir, 'rgb/exp/', txt)
        # bb_anno_t = os.path.join(dir, 't/exp/', txt)
        # if os.path.exists(bb_anno_rgb) and os.path.exists(bb_anno_t):
            # pass
        # else:  # This part is the random sampling, which is as a supplement for detector.
            # x1 = random.randint(100,1181)
            # x2 = random.randint(100,1181)
            # while (abs(x1-x2) < 100) :
                # x2 = random.randint(100,1181)
            # x3 = random.randint(100,1181)
            # while (abs(x1-x3) < 100) or (abs(x2-x3) < 100) :
                # x3 = random.randint(100,1181)
            # x4 = random.randint(100,1181)
            # while (abs(x1-x4) < 100) or (abs(x2-x4) < 100) or (abs(x3-x4) < 100) :
                # x4 = random.randint(100,1181)
            # x5 = random.randint(100,1181)
            # while (abs(x1-x5) < 100) or (abs(x2-x5) < 100) or (abs(x3-x5) < 100) or (abs(x4-x5) < 100) :
                # x5 = random.randint(100,1181)
                    
            # y = random.sample(range(100,541),5)   # [100,541) 
            # gt = torch.Tensor([[float(x1), float(y[0]), 100.0, 100.0],
            #                 [float(x2), float(y[1]), 100.0, 100.0],
            #                 [float(x3), float(y[2]), 100.0, 100.0],
            #                 [float(x4), float(y[3]), 100.0, 100.0],
            #                 [float(x5), float(y[4]), 100.0, 100.0]])
            # return gt
        # file_rgb = open(bb_anno_rgb)
        # file_t = open(bb_anno_t)
        # gt_rgb = file_rgb.readlines()
        # gt_t = file_t.readlines()
        # str to float
        # n=0
        # for rgb in gt_rgb:
            # xx = rgb.split()
            # for i in range(6):
                # xx[i] = float(xx[i])
            # gt_rgb[n] = xx
            # n = n+1
        # m=0
        # for t in gt_t:
            # yy = t.split()
            # for j in range(6):
                # yy[j] = float(yy[j])
            # gt_t[m] = yy
            # m = m+1
        # IOU 
        # for rgb in gt_rgb:
            # for t in gt_t:
                # rec1 = rgb[2:]
                # rec2 = t[2:]
                # left_column_max  = max(rec1[0],rec2[0])
                # right_column_min = min(rec1[2],rec2[2])
                # up_row_max       = max(rec1[1],rec2[1])
                # down_row_min     = min(rec1[3],rec2[3])
                # 两矩形无相交区域的情况
                # if left_column_max>=right_column_min or down_row_min<=up_row_max:
                    # iou = 0
                # 两矩形有相交区域的情况
                # else:
                    # S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
                    # S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
                    # S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
                    # iou = S_cross/(S1+S2-S_cross)
                # if iou > 0.8: # 删除重复的框
                    # if rgb[1] > t[1]:
                        # t[1] = 0
                    # else:
                        # rgb[1] = 0
        # for t in gt_t: # T: x+640
            # t[2] = t[2] + 640
            # t[4] = t[4] + 640
        # gt = gt_rgb + gt_t
        # gt = sorted(gt, key=lambda x: x[0], reverse=True)
        # type balance
        # i = 0
        # j = 0
        # person = []
        # other = []
        # for s in gt: # split person and other
            # if s[0] == 0:
                # person = person + [s]
                # i = i+1
            # else:
                # other = other + [s]
                # j = j+1
        # person = sorted(person, key=lambda x: x[1], reverse=True)
        # other = sorted(other, key=lambda x: x[1], reverse=True)
        # i = 0
        # gtt = []  # 2 other 3+ person
        # for s in other:
            # gtt = gtt + [s]
            # i = i+1
            # if i>1:
                # break
        # for t in person:
            # gtt = gtt + [t]
            # i = i+1
            # if i>4:
                # break
        # i = 0
        # for m in gtt:
            # gtt[i] = m[2:]
            # i = i+1
        # m = 0
        # for x in gtt: # <=5
            # x[2] = x[2]-x[0] # w
            # x[3] = x[3]-x[1] # h
            # gtt[m] = x 
            # m = m+1
        # while m < 5:
            # for s in gtt:
                # if m >= 5:
                    # break
                # gtt = gtt + [s]
                # m = m+1
        # return torch.Tensor(gtt) 
        

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_frame(self, seq_path): # TIR
        img = self.image_loader(seq_path)
        return img  #512*1280*3

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        return {'bbox': bbox, 'valid': valid}

    def get_frames(self, seq_id, anno=None):
        img_path = self._get_sequence_path(seq_id)
        img = self._get_frame(img_path) 
        # jointly train RGB-TIR, TIR-RGB
        img = np.resize(img, (512,1280,3)) 
        img1 = img[:,:640,:]
        img2 = img[:,640:,:]
        imgt = np.concatenate((img2,img1),axis=1)
        # train of single branch
        # img = img1  # RGB
        # imgt = img2 # TIR
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...].clone()]
        return img, imgt, anno_frames