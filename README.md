Image-to-image translation by CNNs trained on paired data (AUTOMAP + Pix2pix)
=============
By Shizuo KAJI

This is an implementation of image-to-image translation using a paired image dataset.
An encoder-decoder network with a discriminator (conditional GAN) is used.
Fully-convolutional encoder-decoder network is precomposed with fully-connected layers
to make the receptive field the whole domain so that it is capable of learning image translations involving global information such as the radon transform (CT reconstruction).

In other words, it is a combination of AUTOMAP

- Image reconstruction by domain transform manifold learning,
Zhu B, Liu JZ, Cauley SF, Rosen BR, Rosen MS,
Nature. 555(7697):487-492, 2018

and Pix2pix
- Image-to-Image Translation with Conditional Adversarial Networks,
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, CVPR, 2017

with some improvements.

This code is based on 
- https://github.com/naoto0804/chainer-cyclegan
- https://github.com/pfnet-research/chainer-pix2pix


## Licence
MIT Licence

## Requirements
- chainer >= 5.3.0: install with 

```pip install cupy,chainer,chainerui,chainercv```

## Usage
```python train_cgan.py -t ct_reconst_train.txt --val ct_reconst_val.txt -R ~/radon -o ~/result -g 0 -e 200 -gfc 1 -u none -l1 0 -l2 1.0 -ldis 0.1 -ltv 0.01 -cw 112 -ch 112```

Learns translation of images in ct_reconst_train.txt placed under "radon/" and outputs the result under "result/". The images are cropped to 112x112.
Train the model with GPU (id 0) and 200 epochs.
Use the network with 1 FC layers at the beginning (as in AUTOMAP) without UNet-like skip connections.
The loss consists of L2 reconstruction error, discriminator, and total variation with weights 1.0 0.1 0.01 respectively.

Sample images can be created by create_shapes_sinogram.py.

For a list of command-line arguments,

```python train_cgan.py -h```
