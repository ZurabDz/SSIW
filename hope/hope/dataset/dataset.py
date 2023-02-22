from torch.utils.data import Dataset
import torch.nn as nn
import cv2
from glob import glob
import os
import os.path as osp
from hope.utils.xml_utils import parse_cmp_xml


class CMPFacadeDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.xmls = glob(osp.join(self.root_dir, '*.xml'))

    def __len__(self):
        return len(self.xmls)

    def __getitem__(self, index):
        metadatas = parse_cmp_xml(self.xmls[index])
        filename = os.path.basename(self.xmls[index].spli('.')[0])
        photo_path = osp.join(self.root_dir, filename)
        rgb = cv2.imread(photo_path, -1)[:, :, ::-1]

        return rgb, metadatas