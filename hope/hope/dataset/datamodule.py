import pytorch_lightning as pl
from .dataset import CMPFacadeDataset
from torch.utils.data import DataLoader
from typing import Union, Optional
from pathlib import Path
# from .collate import MyCollator


class CMPFacadeDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: Optional[int] = 4, num_workers: Optional[int] = 4):
        super().__init__()
        if isinstance(root_dir, Path):
            self.root_dir = str(root_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cmp_facade_train = CMPFacadeDataset(root_dir)
        # self.custom_collate_fn = MyCollator(root_dir)

    def train_dataloader(self):
        # return DataLoader(self.cmp_facade_train, batch_size=self.batch_size,
        #                   num_workers=self.num_workers, collate_fn=self.custom_collate_fn, shuffle=True)
        return DataLoader(self.cmp_facade_train, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)
