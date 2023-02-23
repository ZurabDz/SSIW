from torch.utils.data import Dataset
import cv2
from glob import glob
import os
import os.path as osp
from hope.utils.xml_utils import parse_cmp_xml
import torch
from hope.utils.transform_utils import (
    resize_by_scaled_short_side, get_imagenet_mean_std, pad_to_crop_sz, normalize_img)
import numpy as np
import json


class CMPFacadeDataset(Dataset):
    def __init__(self, root_dir: str, class_definitions=None) -> None:
        self.root_dir = root_dir
        self.xmls = glob(osp.join(self.root_dir, '*.xml'))
        with open(class_definitions, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.index_to_label = {}
        self.label_to_index = {}

        for i, key in enumerate(data.keys(), start=0):
            self.index_to_label[i] = key
            self.label_to_index[key] = i

        self.num_class = len(self.index_to_label)  # 0th is unkown

    def __generate_mask(self, metadata, height, width):
        mask = torch.zeros((self.num_class, height, width))
        mask[self.label_to_index['unlabeled']].fill_(1)
        for obj_meta in metadata:
            x1 = round(obj_meta.points.x1 * height)
            x2 = round(obj_meta.points.x2 * height)
            y1 = round(obj_meta.points.y1 * width)
            y2 = round(obj_meta.points.y2 * width)
            mask[self.label_to_index[obj_meta.label_name]][x1:x2+1, y1:y2+1] = 1
            mask[self.label_to_index['unlabeled']][x1:x2+1, y1:y2+1] = 0

        return mask

    def single_scale_single_crop(self, image):
        ori_h, ori_w, _ = image.shape
        mean, std = get_imagenet_mean_std()
        crop_h = (np.ceil((ori_h - 1) / 32) * 32).astype(np.int32)
        crop_w = (np.ceil((ori_w - 1) / 32) * 32).astype(np.int32)

        image, pad_h_half, pad_w_half = pad_to_crop_sz(
            image, crop_h, crop_w, mean)
        image_crop = torch.from_numpy(image.transpose((2, 0, 1))).float()
        normalize_img(image_crop, mean, std)
        image_crop = image_crop.unsqueeze(0)

        return image_crop

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, index):
        metadatas = parse_cmp_xml(self.xmls[index])
        filename = os.path.basename(self.xmls[index].split('.')[0])
        photo_path = osp.join(self.root_dir, filename + '.jpg')
        rgb = cv2.imread(photo_path, -1)[:, :, ::-1]
        # TODO: move 720
        image_resized = resize_by_scaled_short_side(rgb, 720, 1)
        # TODO: return initial size
        image_crop = self.single_scale_single_crop(image_resized)
        _, _, height, width, = image_crop.shape
        mask = self.__generate_mask(metadatas, height, width)
        return image_crop, mask
