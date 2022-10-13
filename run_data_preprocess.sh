#!/bin/bash

python chalearn_sample_data.py
python chalearn_video_to_images.py
python chalearn_image_to_padded.py
python chalearn_padded_to_iuv.py
python chalearn_iuv_to_crop.py