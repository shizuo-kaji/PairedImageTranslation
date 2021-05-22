#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import cv2
import os
import numpy as np
from numpy.random import *
import pydicom as dicom
from chainercv.utils import write_image,read_image
import glob
from skimage.morphology import remove_small_objects

#########################
def main():
    parser = argparse.ArgumentParser(description='create sinograms for artificial images')
    parser.add_argument('--size', '-s', type=int, default=64,
                        help='minimum area')
    parser.add_argument('--threshold', '-t', type=int, default=200,
                        help='threshold 0--256')
    parser.add_argument('--maskdir', '-m', default='mask',
                        help='directory containing mask JPEG files')
    parser.add_argument('--imagedir', '-i', default='images',
                        help='directory containing DCM files')
    parser.add_argument('--outdir', '-o', default='out',
                        help='directory to output masked DCM files')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ###
    for file in sorted(glob.glob(args.imagedir+"/*.dcm", recursive=False)):
        print(file)
        fn,ext = os.path.splitext(file)
        bfn = os.path.basename(fn)

        ref_dicom = dicom.read_file(file, force=True)
        ref_dicom.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
        dt=ref_dicom.pixel_array.dtype
        dat = ref_dicom.pixel_array.astype(np.float32) +ref_dicom.RescaleIntercept
        print(dat.shape,np.min(dat),np.mean(dat),np.max(dat))

        imgfn=os.path.join(args.maskdir,bfn+".jpg")
        print(imgfn)
        mask = read_image(imgfn)[0] < args.threshold
        mask = remove_small_objects(mask,min_size=args.size,in_place=True)
        mask = ~remove_small_objects(~mask,min_size=args.size)

        dat[mask] = -2048
        dat -= ref_dicom.RescaleIntercept
        dat = dat.astype(dt)           
        ref_dicom.PixelData = dat.tostring()
        ref_dicom.save_as(os.path.join(args.outdir,bfn+".dcm"))



if __name__ == '__main__':
    main()
