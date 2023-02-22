from torch.utils.data import Dataset
import torch.nn as nn
import cv2
from glob import glob
import os
import os.path as osp
from hope.utils.xml_utils import parse_cmp_xml
import torch


class CMPFacadeDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.xmls = glob(osp.join(self.root_dir, '*.xml'))
        self.num_class = 13 # 0th is unkown


    def __generate_mask(self, metadata, height, width):
        mask = torch.zeros((self.num_class + 1, height, width))
        mask[0].fill_(1)
        for obj_meta in metadata:
            x1 = round(obj_meta.points.x1 * height)
            x2 = round(obj_meta.points.x2 * height)
            y1 = round(obj_meta.points.y1 * width)
            y2 = round(obj_meta.points.y2 * width)
            mask[obj_meta.label][x1:x2+1, y1:y2+1] = 1
            mask[0][x1:x2+1, y1:y2+1] = 0

        return mask

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, index):
        metadatas = parse_cmp_xml(self.xmls[index])
        filename = os.path.basename(self.xmls[index].split('.')[0])
        photo_path = osp.join(self.root_dir, filename + '.jpg')
        rgb = cv2.imread(photo_path, -1)[:, :, ::-1]
        height, width, _ = rgb.shape
        mask = self.__generate_mask(metadatas, height, width)
        return rgb, mask