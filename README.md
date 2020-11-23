# SpiralNet

This repository provides the official PyTorch implementation of our paper "Spiral Generative Network for Image Extrapolation".

Our paper can be found in https://link.springer.com/chapter/10.1007/978-3-030-58529-7_41.


## Prerequisites

- Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN

## Getting Started


### Installation

- Clone this repo:
```bash
git clone https://github.com/zhenglab/spiralnet.git
cd spiralnet
```

- Install [PyTorch](http://pytorch.org) and 1.0+ and other dependencies (e.g., torchvision).
  - For pip users, please type the command `pip install -r requirement.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yaml`.

### ImagineGAN 

- Training

```
python train.py --path=$configpath$

For example: python train.py --path=./checkpoints/ImagineGAN/celeba/
```

- Testing

```
python test.py --path=$configpath$ 

For example: python test.py --path=./checkpoints/ImagineGAN/celeba/
```


### SliceGAN

Put the ImagineGAN model in the corresponding directory, for example,checkpoints/SliceGAN/celeba/imagine_g.pth. 

- Training

```
python train.py --path=$configpath$

For example: python train.py --path=./checkpoints/SliceGAN/celeba/
```

- Testing

```
python test.py --path=$configpath$ 

For example: python test.py --path=./checkpoints/SliceGAN/celeba/
```


## Citing
```
@inproceedings{guo2020spiralnet,
author = {Guo, Dongsheng and Liu, Hongzhi and Zhao, Haoru and Cheng, Yunhao and Song, Qingwei and Gu, Zhaorui and Zheng, Haiyong and Zheng, Bing},
title = {Spiral Generative Network for Image Extrapolation},
booktitle = {The European Conference on Computer Vision (ECCV)},
pages={701--717},
year = {2020}
} 

```
