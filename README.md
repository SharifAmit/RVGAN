# MICCAI2021 RVGAN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rv-gan-retinal-vessel-segmentation-from/retinal-vessel-segmentation-on-drive)](https://paperswithcode.com/sota/retinal-vessel-segmentation-on-drive?p=rv-gan-retinal-vessel-segmentation-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rv-gan-retinal-vessel-segmentation-from/retinal-vessel-segmentation-on-chase_db1)](https://paperswithcode.com/sota/retinal-vessel-segmentation-on-chase_db1?p=rv-gan-retinal-vessel-segmentation-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rv-gan-retinal-vessel-segmentation-from/retinal-vessel-segmentation-on-stare)](https://paperswithcode.com/sota/retinal-vessel-segmentation-on-stare?p=rv-gan-retinal-vessel-segmentation-from)


This code is for our paper "RV-GAN: Segmenting Retinal Vascular Structure inFundus Photographs using a Novel Multi-scaleGenerative Adversarial Network" which is part of the supplementary materials for MICCAI 2021 conference. The paper has since been accpeted to MICCAI 2021 and will be presented in September 2021.

![](img1.png)

### Arxiv Pre-print
```
https://arxiv.org/pdf/2101.00535v2.pdf
```

# Citation 
```
@article{kamran2021rv,
  title={RV-GAN: Segmenting Retinal Vascular Structure in Fundus Photographs using a Novel Multi-scale Generative Adversarial Network},
  author={Kamran, Sharif Amit and Hossain, Khondker Fariha and Tavakkoli, Alireza and Zuckerbrod, Stewart Lee and Sanders, Kenton M and Baker, Salah A},
  journal={arXiv preprint arXiv:2101.00535v2}, 
  year={2021}
}
```

## Pre-requisite
- Ubuntu 18.04 / Windows 7 or later
- NVIDIA Graphics card

## Installation Instruction for Ubuntu
- Download and Install [Nvidia Drivers](https://www.nvidia.com/Download/driverResults.aspx/142567/en-us)
- Download and Install via Runfile [Nvidia Cuda Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
- Download and Install [Nvidia CuDNN 7.6.5 or later](https://developer.nvidia.com/rdp/cudnn-archive)
- Install Pip3 and Python3 enviornment
```
sudo apt-get install pip3 python3-dev
```
- Install Tensorflow-Gpu version-2.0.0 and Keras version-2.3.1
```
sudo pip3 install tensorflow-gpu==2.0.3
sudo pip3 install keras==2.3.1
```
- Install packages from requirements.txt
```
sudo pip3 install -r requirements.txt
```
## Work in Progress
