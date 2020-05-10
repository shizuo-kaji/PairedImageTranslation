import os
import glob
import pandas as pd 
import numpy as np
from chainercv.transforms import resize
from chainercv.utils import read_image,write_image
import pydicom as dicom

import argparse

def img2var(self,img):
    # cut off mask [-1,1] or [0,1] output
    return(2*(np.clip(img,-1024,1024)+1024)/2048-1.0)

parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
parser.add_argument('--root', '-R', help='input dir containing images')
parser.add_argument('--out', '-o', help='output dir')
parser.add_argument('--noise', '-n', default=100, type=float, help='strength of Poisson noise')
parser.add_argument('--imgtype', '-it', default="jpg", help="image file type (file extension)")
args = parser.parse_args()

os.makedirs(os.path.join(args.out,"trainA"), exist_ok=True)
os.makedirs(os.path.join(args.out,"trainB"), exist_ok=True)

for fullname in sorted(glob.glob(os.path.join(args.root,"**/*.{}".format(args.imgtype)), recursive=True)):
    fn = os.path.basename(fullname)
    fn,ext = os.path.splitext(fn)
    if args.imgtype == 'dcm':
        subdirname = os.path.basename(os.path.dirname(fullname))
        ref_dicom_in = dicom.read_file(fullname, force=True)
        ref_dicom_in.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
        dt=ref_dicom_in.pixel_array.dtype
        fileB = "trainB/{}_{}_clean.dcm".format(subdirname,fn)
        ref_dicom_in.save_as(os.path.join(args.out,fileB))
        dat = ref_dicom_in.pixel_array
        print(np.min(dat),np.max(dat))
        # noise
        dat = (ref_dicom_in.pixel_array + np.random.poisson(args.noise,ref_dicom_in.pixel_array.shape)).astype(dt)
        print(np.min(dat),np.max(dat))
        ref_dicom_in.PixelData = dat.tostring()
        fileA = "trainA/{}_{}_noise.dcm".format(subdirname,fn)
        ref_dicom_in.save_as(os.path.join(args.out,fileA))
    else:
        dat = read_image(fullname)
        c,h,w = dat.shape
        fileB = "trainB/{}_clean.jpg".format(fn)
        write_image(dat,os.path.join(args.out,fileB))
        dat += np.random.poisson(args.noise,dat.shape)
        fileA = "trainA/{}_noise.jpg".format(fn)
        write_image(dat,os.path.join(args.out,fileA))
    print("{}\t{}".format(fileA,fileB))    