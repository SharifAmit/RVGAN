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
### Dataset Pre-processing

- Type this in terminal to run the **strided_crop_DRIVE.py**, **strided_crop_STARE.py** or **strided_crop_CHASE.py** file. 
```
python3 strided_crop_DRIVE.py --input_dim=128 --stride=32
```
- There are different flags to choose from. Not all of them are mandatory.
```
    '--input_dim', type=int, default=128
    '--stride', type=int, default=32
```

### NPZ file conversion
- Convert all the images to npz format using **convert_npz_DRIVE.py**, **convert_npz_STARE.py** or **convert_npz_CHASE.py** file. 
```
python3 convert_npz_DRIVE.py --input_dim=(128,128) --n_crops=210 --outfile_name='DRIVE_stride_32_dim_128'
```
- There are different flags to choose from. Not all of them are mandatory.
```
    '--input_dim', type=int, default=(128,128)
    '--n_crops', type=int, default=210
    '--outfile_name', type=str, default='DRIVE_stride_32_dim_128'
```

## Training

- Type this in terminal to run the train.py file
```
python3 train.py --npz_file=DRIVE --batch=4 --epochs=200 --savedir=RVGAN
```
- There are different flags to choose from. Not all of them are mandatory

```
   '--npz_file', type=str, default='attenton2angio', help='path/to/npz/file'
   '--batch_size', type=int, default=4
   '--input_dim', type=int, default=128
   '--epochs', type=int, default=200
   '--savedir', type=str, required=False, help='path/to/save_directory',default='RVGAN'
```

# License
The code is released under the BSD-3 License, you can read the license file included in the repository for details.

## Work in Progress
