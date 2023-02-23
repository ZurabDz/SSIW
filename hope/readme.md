## Here we are trying to finetune existing checkpoint on CMP dataset 

#### Steps for running training:

- `conda create -n new_env python=3.8`
- `git clone "repo"`
- `cd "repo"/ && pip install .`
- `cd scripts/ && python download_stuff.py`
- `bash train.bash`