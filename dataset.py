# -*- coding: utf-8 -*-

import numpy
from PIL import Image
import six
import numpy as np
import os
import random

from chainer.dataset import dataset_mixin
from chainercv.transforms import resize,random_flip,random_crop
from chainercv.utils import read_image,write_image

def stack_imgs(fns,crop,resize=False,grey=False):
    imgs_in =[]
    for fn in fns:
        fn1,ext = os.path.splitext(fn)
        # image can be given as csv or jpg/png... etc
        if ext==".csv":
            img_in = np.loadtxt(fn, delimiter=",")[np.newaxis,]
        else:
            img_in = read_image(fn, color=not grey)/127.5 -1.0
        # resize if the image is too small
        if resize:
            if img_in.shape[1]<crop[0] or img_in.shape[2]<crop[1]:
                if crop[0]/img_in.shape[1] < crop[1]/img_in.shape[2]:
                    img_in = resize(img_in, (int(crop[1]/img_in.shape[2]*img_in.shape[1]), crop[1]))
                else:
                    img_in = resize(img_in, (crop[0], int(crop[0]/img_in.shape[1]*img_in.shape[2])))
        imgs_in.append(img_in)
    # an input/output image can consist of multiple images; they are stacked as channels
#    print(imgs_in.shape)
    return(np.concatenate(imgs_in, axis=0))


class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, datalist, DataDir, from_col, to_col, crop=None, random=False, grey=False):
        self.dataset = []
        ## an input/output image can consist of multiple images; they are stacked as channels
        with open(datalist) as input:
            for line in input:
                files = line.strip().split('\t')
                if(len(files)<len(set(from_col).union(set(to_col)))):
                    print("Error in reading data file: ",files)
                    exit()
                self.dataset.append([
                    [os.path.join(DataDir,files[i]) for i in from_col],
                    [os.path.join(DataDir,files[i]) for i in to_col]
                ])
                for i in range(len(files)):
                    if not os.path.isfile(os.path.join(DataDir,files[i])):
                        print("{} not found!".format(os.path.join(DataDir,files[i])))
                        exit()
        self.crop = crop
        self.grey = grey
        self.random = random # random crop/flip for data augmentation
        print("loaded {} images".format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    def get_img_path(self, i):
        return '{:s}'.format(self.dataset[i][0][0])

    def var2img(self,var):
        return(0.5*(1.0+var)*255)

    def get_example(self, i):
        il,ol = self.dataset[i]
        imgs_in = stack_imgs(il,self.crop, grey=self.grey)
        imgs_out = stack_imgs(ol,self.crop, grey=self.grey)
        H, W = self.crop
        if self.random: # random crop/flip
            if random.choice([True, False]):
                imgs_in = imgs_in[:, :, ::-1]
                imgs_out = imgs_out[:, :, ::-1]
            y_offset = random.randint(0, max(imgs_in.shape[1] - H,0))
            y_slice = slice(y_offset, y_offset + H)
            x_offset = random.randint(0, max(imgs_in.shape[2] - W,0))
            x_slice = slice(x_offset, x_offset + W)                
        else: # centre crop
            y_offset = int(round((imgs_in.shape[1] - H) / 2.))
            x_offset = int(round((imgs_in.shape[2] - W) / 2.))
            y_slice = slice(y_offset, y_offset + H)
            x_slice = slice(x_offset, x_offset + W)
        return imgs_in[:,y_slice,x_slice], imgs_out[:,y_slice,x_slice]
    
