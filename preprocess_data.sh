#!/bin/bash

conda activate focalpose

python -m focalpose.scripts.cars_to_urdf
python -m focalpose.scripts.pix3d_to_urdf
python -m focalpose.scripts.preprocess_cars --dataset stanfordcars3d
python -m focalpose.scripts.preprocess_cars --dataset compcars3d
python -m focalpose.scripts.patch_uv