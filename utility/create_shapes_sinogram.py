#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import cv2
import os
import numpy as np
from numpy.random import *

import matplotlib
matplotlib.use('Agg')

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon
from scipy.ndimage import zoom

def MakeRectangle(s):
    img1 = np.full((s, s), 1, dtype=np.uint8)
    cv2.rectangle(img1, (randint(0,s), randint(0,s)), (randint(0,s), randint(0,s)), randint(s/5,s),-1)
    rad = np.random.rand()*2*np.pi
    matrix =[[1,np.tan(rad),0],[0,1,0]]
    affine_matrix = np.float32(matrix)
    return cv2.warpAffine(img1, affine_matrix, (s,s), flags=cv2.INTER_LINEAR)
def MakeEllipse(s):
    img1 = np.full((s, s), 1, dtype=np.uint8)
    cv2.ellipse(img1, (randint(0,s), randint(0,s)), (randint(0,s/2), randint(0,s/2)), 0, 0, 360, randint(s/5,s), -1)
    rad = np.random.rand()*2*np.pi
    matrix =[[1,np.tan(rad),0],[0,1,0]]
    affine_matrix = np.float32(matrix)
    return cv2.warpAffine(img1, affine_matrix, (s,s), flags=cv2.INTER_LINEAR)
def MakeImage(s):
    if randint(1,3)==1:
        return MakeRectangle(s)
    else:
        return MakeEllipse(s)

#########################
def main():
    parser = argparse.ArgumentParser(description='create sinograms for artificial images')
    parser.add_argument('--size', '-s', type=int, default=128,
                        help='size of the image')
    parser.add_argument('--num', '-n', type=int, default=2000,
                        help='Number of images to be created')
    parser.add_argument('--noise', '-z', type=int, default=10,
                        help='Strength of noise')
    parser.add_argument('--outdir', '-o', default='radon',
                        help='output directory')
    args = parser.parse_args()

    ###
    os.makedirs(args.outdir, exist_ok=True)
    dir_origin = "original"
    dir_sinogram = "sinogram"
    os.makedirs(os.path.join(args.outdir,dir_origin), exist_ok=True)
    os.makedirs(os.path.join(args.outdir,dir_sinogram), exist_ok=True)
    fn_origin, fn_sinogram=[], []
    for i in range(args.num):
        img = np.full((args.size, args.size), 1, dtype=np.uint8)
        for j in range(np.random.randint(5,10)):
            img2=MakeImage(args.size)
            img=cv2.addWeighted(img,1,img2,1,0)
        # masking to a circle
        mask = np.zeros((args.size, args.size), dtype=np.uint8)
        cv2.circle(mask, center=(args.size // 2, args.size // 2), radius=args.size//2, color=255, thickness=-1)
        img = np.where(mask==255, img, 0)
        # original image
        fn_origin.append(os.path.join(dir_origin,"s{0:04d}.png".format(i)))
        cv2.imwrite(os.path.join(args.outdir,fn_origin[-1]), img)
        print("original #{}, min {}, max {}".format(i,np.min(img),np.max(img),img.shape))
        # radon transform
        theta = np.linspace(0., 180., num=args.size, endpoint=False)
        img = radon(img, theta=theta, circle=True)
        img = 255*(img/(2*args.size*args.size) )
        print("radon #{}, min {}, max {}".format(i,np.min(img),np.max(img),img.shape))
        fn_sinogram.append(os.path.join(dir_sinogram,"r{0:04d}.png".format(i)))
        cv2.imwrite(os.path.join(args.outdir,fn_sinogram[-1]), np.clip(img,0,255).astype(np.uint8))
        # add noise
#        img = np.clip(img+np.random.randint(-args.noise,args.noise,img.shape),0,255)
#        print("radon w/ noise #{}, min {}, max {}".format(i,np.min(img),np.max(img),img.shape))
#        cv2.imwrite(os.path.join(args.outdir,"nn{0:04d}.png".format(i)), img)
        # reconstructed by inverse radon transform
#        reconstruction = iradon(img/256 * 2*args.size*args.size, theta=theta, circle=True)
#        cv2.imwrite(os.path.join(args.outdir,"i{0:04d}.png".format(i)), reconstruction)

    ### file list
    n=int(args.num*0.8)
    with open(os.path.join(args.outdir,"ct_reconst_train.txt"), "w") as f:
        for i in range(n):
            f.write("{}\t{}\n".format(fn_origin[i],fn_sinogram[i]))
    with open(os.path.join(args.outdir,"ct_reconst_val.txt"), "w") as f:
        for i in range(n,args.num):
            f.write("{}\t{}\n".format(fn_origin[i],fn_sinogram[i]))

if __name__ == '__main__':
    main()
