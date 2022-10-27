#!/bin/bash
# sudo apt install gcc g++ ninja-build
# python -m pip install -e detectron2
# cd pyflow
# pip install cython
# python setup.py build_ext -i

python chalearn_sample_data.py
python chalearn_video_to_images.py
python chalearn_image_to_padded.py
python chalearn_padded_to_iuv.py
python chalearn_video_to_flow.py
python chalearn_iuv_to_crop.py
python train.py