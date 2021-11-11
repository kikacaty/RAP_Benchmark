#!/bin/bash
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
pip install cityscapesscripts
pip install -e . --user

pip install kornia==0.5.1

apt-get update
yes | apt install libgl1-mesa-glx

# prepare dataset
cp /cityscapes/.zip data/cityscapes/
yes | unzip data/cityscapes/gtFine_trainvaltest.zip -d data/cityscapes/
yes | unzip data/cityscapes/leftImg8bit_trainvaltest.zip -d data/cityscapes/

python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8