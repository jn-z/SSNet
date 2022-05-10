import os
import math
import random
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
import gc;

gc.enable()  # memory is tight
import torch
import pdb
import torch.utils.data as data
import cv2
from image import flip, color_aug
from image import get_affine_transform, affine_transform

dtype = "float32"


## HeatMap Genrating Functions

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def reverse_res_to_bbox(res):
    down_ratio = res['input'].shape[1] / res['hm'].shape[1]
    output_h, output_w = res['hm'].shape[0], res['hm'].shape[1]
    num_objs = np.sum(res['reg_mask'])
    bbox = np.zeros((num_objs, 4), dtype='float32')
    ct = res['reg'][:num_objs]
    ct[:, 0] += res['ind'][:num_objs] % output_w
    ct[:, 1] += res['ind'][:num_objs] // output_w
    h, w = res['wh'][:num_objs, 1], res['wh'][:num_objs, 0]
    bbox[:, 0] = (ct[:, 0] * 2 - w) / 2
    bbox[:, 2] = (ct[:, 0] * 2 + w) / 2
    bbox[:, 1] = (ct[:, 1] * 2 - h) / 2
    bbox[:, 3] = (ct[:, 1] * 2 + h) / 2
    bbox *= down_ratio
    return bbox


class ctDataset(data.Dataset):

    def __init__(self,split="train"):
        # img_dir = 'kitti dataset/'
        # ../input/kitti_single/training/label_2/
        self.keep_res = False
        self.split = split
        self.not_rand_crop = True
        self.no_color_aug = False
        self.original_shift = 0
        self.original_scale = 0
        self.scale = [0,0.02,0.05,0.1]
        self.shift = [0,0.02,0.05,0.1]
        # self.scale = [0, 0.7, 0.9, 1.2]
        # self.shift = [0, 0.1,0.2]
        self.flip = 0.5
        self.scale_shift_values = 0.3
        self.num_classes = 3
        self.max_objs = 200
        self.down_ratio = 2
        self.color_aug_value = 0.5
        self.input_h, self.input_w = 512, 512
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
        self.mean = [0.64566372, 0.64566372, 0.64566372]
        self.std = [0.08311639, 0.08311639, 0.08311639]
        root = '/media/zjn/F/Data/TX_data/all_bbox/'
        #pdb.set_trace()
        self.root = root
        self.image_dir = os.path.join(root,"RGB/4K_800_801_data_imag_add_mode/")  # ceshi/")#short_data_imag_4000/")
        self.label_dir = os.path.join(root, "label_txt/4K_800_801_label_txt_add_mode/")
        if self.split == "train":
            imageset_txt = os.path.join(root, "imageSets", "train.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(root, "imageSets", "test.txt")
        elif self.split == "validation":
            imageset_txt = os.path.join(root, "imageSets", "test.txt")
        image_files = []
        for line in open(imageset_txt, "r"):
            image_name = line.replace("\n", "")
            image_files.append(image_name)
            # image_name_2 = image_name.replace("RGB_500","RGB_1000")
            # image_files.append(image_name_2)
            # image_name_3 = image_name.replace("RGB_500", "RGB_2000")
            # image_files.append(image_name_3)
            # image_name_4 = image_name.replace("RGB_500", "RGB_4000")
            # image_files.append(image_name_4)
        self.image_files = image_files
        self.label_files = [i.replace(".jpg", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[1], box[0], box[3], box[2]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        #pdb.set_trace()
        original_idx = self.image_files[idx].replace(".jpg", "")
        img_path = os.path.join(self.image_dir + self.image_files[idx])
        img = cv2.imread(img_path)
        #default_resolution = [1601, 400]  # [375, 1242]
        # height, width = img.shape[0], img.shape[1]
        # print("Before pre-reshape: ", height, width)
        # transform to 512 * 512

        # reshape image to default size (some samples have slightly different size)
        height, width = img.shape[0], img.shape[1]
        cc = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.keep_res:
            input_h = (height | self.pad) + 1
            input_w = (width | self.pad) + 1
            ss = np.array([input_w, input_h], dtype=np.float32)
        else:
            ss = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.input_h, self.input_w
        # pdb.set_trace()
        flipped = False
        if self.split == "train":
            if not self.not_rand_crop:
                ss = ss * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                cc[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                cc[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                if np.random.random() < self.scale_shift_values:
                   sf = np.random.choice(self.scale)
                   cf = np.random.choice(self.shift)
                else:
                    sf = self.original_scale
                    cf = self.original_shift
                cc[0] += ss * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                cc[1] += ss * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                ss = ss * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if np.random.random() < self.flip:
               #pdb.set_trace()
               flipped = True
               img = img[::-1, ::-1, :]
               cc[0] = width - cc[0] - 1
        #pdb.set_trace()
        trans_input = get_affine_transform(cc, ss, 0, [input_w, input_h])
        #mapping = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float)
        inp = cv2.warpAffine(img, trans_input,(input_w, input_h))
        #print(inp)
        #inp = cv2.resize(img, (input_w, input_h))
        # scale_h, scale_w = input_h / height, input_w / width
        inp = (inp.astype(np.float32) / 255.)
        if self.split == "train":
           if np.random.random() < self.color_aug_value:
               color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        #inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        inp = inp.astype(np.float32)
        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(cc, ss, 0, [output_w, output_h])
        img = cv2.resize(img, (width, height))

        # get the labels
        label_path = os.path.join(self.label_dir + self.label_files[idx])
        with open(label_path) as f:
            content = f.readlines()
        content = [x.split() for x in content]
        # print(content)

        draw_gaussian = draw_umich_gaussian
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cls = np.zeros((self.max_objs), dtype=np.float32)
        # con_signal = np.zeros((max_objs), dtype=np.int16)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        #pdb.set_trace()
        count = 0
        for c in content:
            # if (c[0] == "Car"):
            #pdb.set_trace()
            bbox = np.array(c[1:5], dtype="float32")
            bbox = self._coco_box_to_bbox(bbox)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                bbox[[1, 3]] = height - bbox[[3, 1]] - 1
                # bbox_new3 = width - bbox[1] - 1
                # bbox[1] = width - bbox[3] - 1
                # bbox[3] = bbox_new3
                # bbox_new2 = height - bbox[0] - 1
                # bbox[0] = height - bbox[2] - 1
                # bbox[2] = bbox_new2
            cls_id = int(c[5])
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[count] = 1. * w, 1. * h
                ind[count] = ct_int[1] * output_w + ct_int[0]
                reg[count] = ct - ct_int
                reg_mask[count] = 1
                count = count + 1
                cls[count] = cls_id
        res = {'image': img, \
               'input': inp, \
               'hm': hm, \
               'reg_mask': reg_mask, \
               'ind': ind, \
               'wh': wh, \
               'reg': reg, \
               'cl': cls, \
               # 'con_id':con_signal, \
               'index': idx, \
               'ori_index': original_idx}

        return res


if __name__ == "__main__":
    im_idx = 10
    my_dataset = ctDataset(split="train")
    res = my_dataset.__getitem__(im_idx)

    img = res['image']
    inp = res['input'].transpose(1, 2, 0)
    hm = res['hm']
    print(inp.dtype)

    plt.title("Original Image")
    plt.imshow(img)
    plt.show()

    plt.title("transform Image")
    plt.imshow(inp)
    plt.show()

    plt.title("Ground Truth Heat Map")
    im = plt.imshow(hm[0])
    plt.colorbar(im)
    plt.show()

    plt.title("Ground Truth Heat Map")
    im = plt.imshow(hm[1])
    plt.colorbar(im)
    plt.show()

    bbox = reverse_res_to_bbox(res)
    for b in bbox:
        cv2.rectangle(inp, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 1), 2)

    plt.title("Calculated bounding box positions")
    plt.imshow(inp)
    plt.show()