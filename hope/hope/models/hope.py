from .segmodel import SegModel
import pytorch_lightning as pl
from hope.utils.constants import INITIAL_MODEL_PATH, CLASS_DEFINITIONS_PATH
import torch
from hope.utils.embed import create_embs_from_names
import json
from hope.utils.embed import get_prediction
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F


class HopeModel(pl.LightningModule):
    def __init__(self, num_classes: int = 512, initial_model_path: str = INITIAL_MODEL_PATH) -> None:
        super().__init__()
        self.model = SegModel(criterions=None,
                              num_classes=num_classes,
                              load_imagenet_model=False, imagenet_ckpt_fpath='')

        self.model = self.model.eval()
        self.model = torch.nn.DataParallel(self.model)
        checkpoint = torch.load(initial_model_path, map_location='cpu')[
            'state_dict']
        ckpt_filter = {k: v for k, v in checkpoint.items(
        ) if 'criterion.0.criterion.weight' not in k}
        self.maybe_errors = self.model.load_state_dict(
            ckpt_filter, strict=False)

        with open(CLASS_DEFINITIONS_PATH, 'r', encoding='utf-8') as f:
            new_definitions = json.load(f)

        self.user_label = list(new_definitions.keys())

        self.gt_embs_list = create_embs_from_names(
            self.user_label, new_definitions).float()
        self.warmup_steps = 50
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, image_crop, mask):
        with torch.no_grad():
            emb, _, _ = self.model(
                inputs=image_crop, label_space=['universal'])
            logit = get_prediction(emb, self.gt_embs_list)

        return logit, mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-5, weight_decay=0.0004)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10,)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(
            1, step / self.warmup_steps))
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        features, labels = train_batch
        logit, mask = self.forward(features, labels)

        loss = self.criterion(logit, mask)
        self.log('train', loss)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        ...
