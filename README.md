# Zero-Reference Deep Curve Estimation (ZeroDCE) for Low-Light Image Enhancement
PyTorch implementation of [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/pdf/2001.06826.pdf) Chongyi Li et al.

## Execute code
* Copy the https link from GitHub repository.
* Using terminal use git clone <https link> in your desired directory.
* Type cd ZeroDCE in terminal.
* Execute <b>test_one.py</b> script to run pretrained model on a random image.
* Execute <b>train.py</b> to train a new model and save it in the models folder.

## Main Model Architecture
Complete model which will iteratively apply pixel-wise transformations to an image to enhance it.

<p align="center">
<img src="Assets/main_model_architecture.png">
</p>

## CNN Architecture

<p align="center">
<img src="Assets/CNN_model_architecture.png">
</p>

## Loss Functions
* Spatial Consistency Loss
* Exposure Control Loss
* Color Constancy Loss
* Illumination Smoothness Loss

## Prerequisites
* Pytorch
* NumPy
* python 3

## Some Results

<p align="center">
<img src="Assets/result1.png">
</p>

<p align="center">
<img src="Assets/result2.png">
</p>

<p align="center">
<img src="Assets/result3.png">
</p>

<p align="center">
<img src="Assets/result4.png">
</p>

## Contributors:

- [Ankur Chemburkar](https://github.com/Developer-Zer0)
- [Devang Jain](https://github.com/djrobin17)
- [Hashir K](https://github.com/hashirkk07)
