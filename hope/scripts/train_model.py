from hope.dataset.datamodule import CMPFacadeDataModule
from hope.models.hope import HopeModel
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="Train Semantic Segmentation model",
    description="configurtion of model",
)

parser.add_argument('--precision', type=int)
parser.add_argument('--max_epochs', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--root_data', type=str)
parser.add_argument('--original_model_path', type=str, required=True)
parser.add_argument('--label_desc_json_path', type=str, required=True)


args = parser.parse_args()

model = HopeModel(num_classes=512, initial_model_path=args.original_model_path, class_definitions=args.label_desc_json_path)

data_module = CMPFacadeDataModule(args.root_data, batch_size=args.batch_size)

trainer = pl.Trainer(accelerator='gpu', gradient_clip_val=1, max_epochs=args.max_epochs,
                     precision=args.precision, 
                     # strategy='deepspeed'
                     # strategy=DeepSpeedStrategy(
                     #         stage=3,
                     #         offload_optimizer=True,
                     #         offload_parameters=True,
                     #     ),
                     )


trainer.fit(model, datamodule=data_module)