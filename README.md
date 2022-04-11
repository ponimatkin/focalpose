<h1 align="center">
FocalPose: Focal Length and Object Pose Estimation via Render and Compare
</h1>

<div align="center">
<h3>
<a href="http://ponimatkin.github.io">Georgy Ponimatkin</a>,
<a href="http://ylabbe.github.io">Yann Labbé</a>,
 <a href="http://bryanrussell.org">Bryan Russell</a>,
<a href="http://imagine.enpc.fr/~aubrym/">Mathieu Aubry</a>,
<a href="http://www.di.ens.fr/~josef/">Josef Sivic</a>
<br>
<br>
CVPR: Conference on Computer Vision and Pattern Recognition, 2022
<br>
<br>
<a href="">[Paper]</a>
<a href="https://ponimatkin.github.io/focalpose/index.html">[Project page]</a>
<br>
</h3>
</div>
 
## Preparing the environment and data
To prepare the environment run the following commands: 
```
conda env create -n focalpose --file environment.yaml
conda activate focalpose

git clone https://github.com/ylabbe/bullet3.git && cd bullet3 
python setup.py build
python setup.py install
```
As a last step, download [MeshLabServer](https://www.meshlab.net) and set the path to its executable in [config.yaml](config.yaml). 

To download the data run `bash download_data.sh`. This will download all the files 
except for the CompCars and texture datasets. For CompCars, please follow [these](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt) instructions
and download the full `.zip` archive named `CompCars.zip` into `local_data` directory. Same needs to be done for the texture dataset
which can be found at [this link](https://drive.google.com/file/d/1Xg8ODMH0k6EZLYFvQ72FF14uQ7uOitpG/view?usp=sharing). After all files are downloaded, just run
```
bash prepare_data.sh
bash preprocess_data.sh
```
This will prepare and preprocess all the files necessary for the codebase.
 
## Rendering synthetic data
The synthetic data needed for training can be generated via:
```
python -m focalpose.scripts.run_dataset_recording --config CONFIG --local
```
You can see all possible configs in the [run_dataset_recording.py](focalpose/scripts/run_dataset_recording.py) file. Synthetic data for Pix3D chair, CompCars and Stanford Cars 
datasets are split into multiple chunks to reduce possible rendering artifacts due to the large number of meshes. There are 21 chunks for the Pix3D chair, 10 for CompCars and 13 for Stanford Cars. 
The rendering process can be potentially sped-up by running the command without `--local` flag. This will use SLURM backend of the
[dask_jobqueue](https://jobqueue.dask.org) library. You will need to fix config of the `SLURMCluster` in the
[record_dataset.py](focalpose/recording/record_dataset.py) according to your cluster.

## Training and evaluating the models
The model can be trained via the following command:
```
python -m focalpose.scripts.run_pose_training --config pix3d-sofa-coarse-disent-F05p
```
This particular config will train coarse model on Pix3D sofa dataset using disentangled loss and 0.5% of real-to-synth data ratio. As another example, the following command will train
refiner model on the Stanford Cars dataset with 10% of real-to-synth data ratio and using the Huber loss:
```
python -m focalpose.scripts.run_pose_training --config stanfordcars3d-refine-huber-F10p
```
We also provide an example submission scripts for [SLURM](train_slurm.sh) and [PBS](tran_pbs.sh) batch systems.

To evaluate the trained coarse and refiner models run (using provided checkpoints as an example):
```
python -m focalpose.scripts.run_pose_evaluation --dataset pix3d-bed.test \
                                               --coarse-run-id pix3d-bed-coarse-F05p-disent--cvpr2022 \
                                               --refine-run-id pix3d-bed-refine-F05p-disent--cvpr2022 \
                                               --mrcnn-run-id detector-pix3d-bed-real-two-class--cvpr2022 \
                                               --niter 15 
```
The pretrained models are located in the `local_data/experiments` folder, which appears after running the data preparation scripts.

## Running inference on the single image
You can also directly run inference on a given image after running the data preparation scripts via:
```
python -m focalpose.scripts.run_single_image_inference --img path/to/image.jpg \
                                                       --cls class_on_image \
                                                       --niter 15 \
                                                       --topk 15 
```
This will run the inference on an image with the class manually provided to the script. The pose will be refined for 15 
iterations and the script will output top-15 model instances predicted by our instance retrieval pipeline. The ouput will consist
of images with aligned meshes, and `.txt` files containing camera matrix and camera pose.

## Citation
If you use this code in your research, please cite the following paper:

```
@inproceedings{ponimatkin2022focal, 
title= {Focal Length and Object Pose Estimation via Render and Compare}, 
author={G. {Ponimatkin} and Y. {Labbe} and B. {Russell} and M. {Aubry} and J. {Sivic}}, 
booktitle={Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR)}, 
year={2022}} }
```

This project is derived from the original [CosyPose](https://github.com/ylabbe/cosypose/) codebase.
