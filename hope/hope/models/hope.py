from .segmodel import SegModel
import pytorch_lightning as pl
from hope.utils.constants import INITIAL_MODEL_PATH, CLASS_DEFINITIONS_PATH
import torch
from hope.utils.embed import create_embs_from_names
import json


class HopeModel(pl.LightningModule):
    def __init__(self, num_classes: int = 512, initial_model_path: str = INITIAL_MODEL_PATH) -> None:
        self.model = SegModel(criterions=None,
                     num_classes=num_classes,
                     load_imagenet_model=False, imagenet_ckpt_fpath='')

        self.model = self.model.eval()
        self.model = torch.nn.DataParallel(self.model)
        checkpoint = torch.load(initial_model_path, map_location='cpu')['state_dict']
        ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
        maybe_errors = self.model.load_state_dict(ckpt_filter, strict=False)
        user_label = ['background','facade','window','door', 'cornice', 'sill',
         'balcony', 'blind','deco', 'molding', 'pillar', 'shop', 'unlabel']

        with open(CLASS_DEFINITIONS_PATH, 'r', encoding='utf-8') as f:
            new_definitions = json.load(f)

        self.gt_embs_list = create_embs_from_names(user_label, new_definitions).float()
        self.criterion = torch.nn.CrossEntropyLoss()
        

    def forward(self):
        ...

    def configure_optimizers(self):
        ...

    def training_step(self, train_batch, batch_idx):
        ...

    def validation_step(self, valid_batch, batch_idx):
        ...    