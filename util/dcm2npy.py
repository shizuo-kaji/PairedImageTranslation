import os
import glob
import pandas as pd 
import numpy as np
from chainercv.utils import write_image
import pydicom as dicom

outdir = "out/"
infn = "*.dcm"
target = "npy"

os.makedirs(outdir, exist_ok=True)

def img2var(img):
    base = -1000
    rn = 2000
    # cut off mask [-1,1] or [0,1] output
    return(2*(np.clip(img,base,base+rn)-base)/rn-1.0)

for file in sorted(glob.glob(infn, recursive=False)):
    print(file)
    fn,ext = os.path.splitext(file)
    if ext==".dcm":
        ref_dicom_in = dicom.read_file(file, force=True)
        ref_dicom_in.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
        dat = ref_dicom_in.pixel_array.astype(np.float32) +ref_dicom_in.RescaleIntercept
        print(dat.shape,np.min(dat),np.mean(dat),np.max(dat))
        dat = img2var(dat)
    elif ext==".txt":
        dat = np.loadtxt(file)
        dat = dat.astype(np.float32)
        dat = dat/127.5 - 1.0
        print(dat.shape,np.min(dat),np.mean(dat),np.max(dat))
    else:
        print("file not found!")
        exit

    if target=="jpg":
        path="{}{}.jpg".format(outdir,fn)
        write_image(dat, path)
    elif target=="npy":
        path="{}{}.npy".format(outdir,fn)
        np.save(path,dat)
    elif target=="csv":
        path="{}{}.csv".format(outdir,fn)
        np.savetxt(path,dat,fmt="%d,%.5f,%.5f")

