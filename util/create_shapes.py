#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numpy.random import *
import os
import argparse

import matplotlib
matplotlib.use('Agg')

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, iradon
from scipy.ndimage import zoom


def main():
    parser = argparse.ArgumentParser(description='create sinograms for artificial images')
    parser.add_argument('--size', '-s', type=int, default=128,
                        help='size of the image')
    parser.add_argument('--num', '-n', type=int, default=1000,
                        help='Number of images to be created')
    parser.add_argument('--outdir', '-o', default='shapes',
                        help='output directory')
    args = parser.parse_args()

    ###
    os.makedirs(args.outdir, exist_ok=True)
    dir_rect = "rectangle"
    dir_ell = "ellipse"
    os.makedirs(os.path.join(args.outdir,dir_rect), exist_ok=True)
    os.makedirs(os.path.join(args.outdir,dir_ell), exist_ok=True)
    fn_rect, fn_ell=[], []
    for i in range(args.num):
        img = np.full((args.size, args.size), 1, dtype=np.uint8)
        center = (randint(args.size//5,4*args.size//5), randint(args.size//5,4*args.size//5))
        axes = (randint(args.size//10,args.size//4), randint(args.size//10,args.size//4))
        colour = randint(50,128)
        UL = (center[0]-axes[0]//2, center[1]-axes[1]//2)
        DR = (center[0]+axes[0]//2, center[1]+axes[1]//2)

        img1 = cv2.rectangle(img, UL, DR, colour,-1)
        fn_rect.append(os.path.join(dir_rect,"r{0:04d}.png".format(i)))
        cv2.imwrite(os.path.join(args.outdir,fn_rect[-1]), img1)

        img2=cv2.ellipse(img,center,axes, 0, 0, 360, colour, -1)
        fn_ell.append(os.path.join(dir_ell,"e{0:04d}.png".format(i)))
        cv2.imwrite(os.path.join(args.outdir,fn_ell[-1]), img1)


    ### file list
    n=int(args.num*0.8)
    with open(os.path.join(args.outdir,"shapes_train.txt"), "w") as f:
        for i in range(n):
            f.write("{}\t{}\n".format(fn_rect[i],fn_ell[i]))
    with open(os.path.join(args.outdir,"shapes_val.txt"), "w") as f:
        for i in range(n,args.num):
            f.write("{}\t{}\n".format(fn_rect[i],fn_ell[i]))

if __name__ == '__main__':
    main()
