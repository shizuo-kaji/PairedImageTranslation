Image-to-image translation by CNNs trained on paired data (AUTOMAP + Pix2pix)
=============
Written by Shizuo KAJI

This code is based on 
- https://github.com/naoto0804/chainer-cyclegan
- https://github.com/pfnet-research/chainer-pix2pix

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

### Requirements
- a modern GPU
- python 3: [Anaconda](https://anaconda.org) is recommended
- chainer >= 5.3.0, cupy, chainerui, chainercv: install them by
```
pip install cupy,chainer,chainerui,chainercv
```
## Licence
MIT Licence

## Training
- First, prepare input-output paired images (x_1,y_1), (x_2,y_2), (x_3,y_3),...
so that the input image x_i is spatially aligned with the output image y_i.
- Make a text file "ct_reconst_train.txt" consisting of:
```
filename of x_1     filename of y_1
filename of x_2     filename of y_2
...
```
The columns are separated by a tab.
- Make another text file "ct_reconst_val.txt" for validation.
- An example command-line arguments for training is
```python train_cgan.py -t ct_reconst_train.txt --val ct_reconst_val.txt -R radon -o result  -cw 128 -ch 128 --grey -g 0 -e 200 -gfc 1 -u none -l1 0 -l2 1.0 -ldis 0.1 -ltv 0.01```
which learns translation of images in ct_reconst_train.txt placed under "radon/" and outputs the result under "result/". 
The images are cropped to 128 x 128 (-cw 128 -ch 128) and converted to greyscale (--grey).
Train the model with GPU (-g 0) and 200 epochs (-e 200).
Use the network with 1 FC layer (-gfc 1) at the beginning (as in AUTOMAP) without UNet-like skip connections (-u none).
The loss consists of L2 reconstruction error, discriminator, and total variation with weights 1.0 0.1 0.01 respectively.

Sample images of Radon transform can be created by 
```
python create_shapes_sinogram.py
```
which corresponds to the included ct_reconst_train.txt and ct_reconst_val.txt

For a list of command-line arguments,

```python train_cgan.py -h```

### Conversion with a trained model
```
python convert.py -b 10 -a results/args --val ct_reconst_val.txt -R radon -o converted -mg results/gen_200.npz
```
converts image files in ct_reconstruct_val.txt using a learnt model "gen_200.npz" and outputs the result to "converted/".
A larger batch size (-b 10) increases the conversion speed but may consume too much memory.

