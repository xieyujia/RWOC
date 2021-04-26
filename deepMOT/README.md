# ROBOT

This is the official implementation of the MOT code for the paper:

**A Hypergradient Approach to Robust Regression without Correspondence** <br />

**[[Paper](https://openreview.net/pdf?id=l35SB-_raSQ)]** <br />


## Contents
1. [Environment Setup](#environment-setup)
2. [Testing](#testing-models)
3. [Training](#training-models)
4. [Demo](#demo)
5. [Acknowledgement](#acknowledgement)

## Environment setup <a name="environment-setup">
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch> 0.4.1, CUDA 9.0 and CUDA 10.0, GTX 1080Ti, Titan X and RTX Titan GPUs.

**warning: the results can be slightly different due to Pytorch version and CUDA version.**

- Clone the repository 
```
git clone https://github.com/yihongXU/deepMOT.git && cd deepmot
```
**Option 1:**
- Setup python environment
```
conda create -n deepmot python=3.6
source activate deepmot
pip install -r requirements.txt
```
**Option 2:**
we offer Singularity images (similar to Docker) for training and testing.
- Open a terminal
- Install singularity
```
sudo apt-get install -y singularity-container
```
- Download a Singularity image and put it to *deepmot/SingularityImages* <br />
[pytorch_cuda90_cudnn7.simg(google drive)](https://drive.google.com/file/d/1wh5dcb_Z3wusl5yn_-0dWl0fTgYYfSAY/view?usp=sharing) <br />
[pytorch1-1-cuda100-cudnn75.simg(google drive)](https://drive.google.com/file/d/1zvQ03pw8hqm6rU_w6lMrOjNcvcrjzRQ4/view?usp=sharing)<br />
[pytorch_cuda90_cudnn7.simg(tencent cloud)](https://share.weiyun.com/5G3Y1FK) <br />
[pytorch1-1-cuda100-cudnn75.simg(tencent cloud)](https://share.weiyun.com/5sTjg6J)
- Open a new terminal
- Launch a Singularity image
```shell
cd deepmot
singularity shell --nv --bind yourLocalPath:yourPathInsideImage ./SingularityImages/pytorch1-1-cuda100-cudnn75.simg
```
**- -bind: to link a singularity path with a local path. By doing this, you can find data from local PC inside Singularity image;** <br />
**- -nv: use local Nvidia driver.**


## Training <a name="training-models">

- [Setup](#environment-setup) your environment

- Download MOT data
Dataset can be downloaded here: e.g. [MOT17](https://motchallenge.net/data/MOT17/) 

- Put *MOT* dataset into *deepmot/data* and it should have the following structure:
```
            mot
            |-------train
            |    |
            |    |---video_folder1
            |    |   |---det
            |    |   |---gt
            |    |   |---img1
            |    |
            |    |---video_folder2
            ...
            |-------test
            |    |
            |    |---video_folder1
            |    |   |---det
            |    |   |---img1
            ...
```

- Download pretrained SOT model *SiamRPNVOT.model*
SiamRPNVOT.model (from SiamRPN, Li et al., see [Acknowledgement](#acknowledgement)): <br />
[SiamRPNVOT.model(google drive)](https://drive.google.com/drive/folders/1HPreiyWbOhgAxhCtvYvoB8wzt_reKzdW?usp=sharing) or <br />
[SiamRPNVOT.model(tencent cloud)](https://share.weiyun.com/5Fxw6ke)

-Put *SiamRPNVOT.model*  to  *deepmot/pretrained/* folder

- run training code
```
python train_mot.py
```
for more details about parameters, do:
```
python train_mot.py -h
```
The trained models are save by default under *deepmot/saved_models/* folder. <br />
The tensorboard logs are saved by default under *deepmot/logs/train_log/* folder and you can visualize your training process by:
```
tensorboard --logdir=/mnt/beegfs/perception/yixu/opensource/deepMOT/logs/train_log
```
**Note:** 
- you should install *tensorflow* (see [tensorflow installation](https://www.tensorflow.org/install/pip)) in order to visualize your training process.
```
pip install --upgrade tensorflow
```

## Testing <a name="testing-models">

Put all pre-trained models to *deepmot/pretrained/*
- run tracking code
```
python tracking_on_mot.py
```
for more details about parameters, do:
```
python tracking_on_mot.py -h
```
The results are save by default under *deepmot/saved_results/txts/test_folder/*.

- Visualization
After finishing tracking, you can visualize your results by plotting bounding box to images.
```
python plot_results.py
```
the results are save by default under *deepmot/saved_results/imgs/test_folder*

**Note:** 
- we clean the detections with nms and threshold of detection scores. They are saved into numpy array in the folder *deepmot/clean_detections*, if you have trouble opening them, try to add *allow_pickle=True* to *np.load()* function.


## Acknowledgement <a name="Acknowledgement">
Some codes are modified and network pretrained weights are obtained from the following repositories: <br />
**Code skeleton**: [**DeepMOT**](https://github.com/yihongXU/deepMOT)
**Single Object Tracker**: [**SiamRPN**](https://github.com/foolwood/DaSiamRPN), [**Tracktor**](https://github.com/phil-bergmann/tracking_wo_bnw/tree/master/src/tracktor).
```
@misc{xu2019train,
    title={How To Train Your Deep Multi-Object Tracker},
    author={Yihong Xu and Aljosa Osep and Yutong Ban and Radu Horaud and Laura Leal-Taixe and Xavier Alameda-Pineda},
    year={2019},
    eprint={1906.06618},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{Zhu_2018_ECCV,
  title={Distractor-aware Siamese Networks for Visual Object Tracking},
  author={Zhu, Zheng and Wang, Qiang and Bo, Li and Wu, Wei and Yan, Junjie and Hu, Weiming},
  booktitle={European Conference on Computer Vision},
  year={2018}
}

@InProceedings{Li_2018_CVPR,
  title = {High Performance Visual Tracking With Siamese Region Proposal Network},
  author = {Li, Bo and Yan, Junjie and Wu, Wei and Zhu, Zheng and Hu, Xiaolin},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}

@InProceedings{tracktor_2019_ICCV,
  author = {Bergmann, Philipp and Meinhardt, Tim and Leal{-}Taix{\'{e}}}, Laura},
  title = {Tracking Without Bells and Whistles},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}}
```
**MOT Metrics in Python**: [**py-motmetrics**](https://github.com/cheind/py-motmetrics)<br />
**Appearance Features Extractor**: [**DAN**](https://github.com/shijieS/SST)<br />
```
@article{sun2018deep,
  title={Deep Affinity Network for Multiple Object Tracking},
  author={Sun, ShiJie and Akhtar, Naveed and Song, HuanSheng and Mian, Ajmal and Shah, Mubarak},
  journal={arXiv preprint arXiv:1810.11780},
  year={2018}
}
```
Training and testing Data from: <br />
**MOT Challenge**: [**motchallenge**](https://motchallenge.net/data)
```
@article{MOT16,
	title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
	shorttitle = {MOT16},
	url = {http://arxiv.org/abs/1603.00831},
	journal = {arXiv:1603.00831 [cs]},
	author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
	month = mar,
	year = {2016},
	note = {arXiv: 1603.00831},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}
}
```
