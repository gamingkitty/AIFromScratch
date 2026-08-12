# AI From Scratch
This GitHub repository contains my attempts at creating various machine learning models from scratch using primarily only numpy/cupy. It contains the code from my first attempt at creating a neural network to computer vision and transformer models!

Most of the code for the neural networks themselves are in the package scratch_model, particularly in layers.py. It contains various layer type classes that contain the code for their forward and backwards passes:
- Dense
- Convolution
- Recurrent
- Attention
- LayerNorm

As well as many more random layer types! A lot of the code is messy and some is incomplete because I'm still working on it and making whatever projects I feel like.

Some of the projects I've attempted so far include:
- Training basic dense models on MNSIT
- Training convolution models to around 85% accuracy on CIFAR-10 and 60% accuracy on CIFAR-100
- Training a 25m parameter model on the tinychat dataset
- Pre-training a 50m parameter transformer model on FineWeb-Edu (About 2.98 validation loss with vocab size 12000 after about 2.4b tokens of training)
- Attempting to make a YOLO model (still a work in progress)
