# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os,glob
import random

from chainer.dataset import dataset_mixin
from chainercv.transforms import resize,random_flip,random_crop
from chainercv.utils import read_image,write_image

try:
    import pydicom as dicom
except:
    pass


class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, datalist, DataDir, from_col, to_col, clip=(None,None), class_num=0, crop=(None,None), imgtype='jpg', random=0, grey=False, BtoA=False, **kwargs):
        self.dataset = []
        self.clip = [None,None]
        if clip[0] is not None:
            self.clip[0] = float(clip[0])
        if clip[1] is not None:
            self.clip[1] = float(clip[1])
        if imgtype=='dcm':
            if clip[0] is None:
                self.clip[0] = -1024
            if clip[1] is None:
                self.clip[1] = 2000

        self.class_num=class_num
        if datalist == '__train__':
            for fn in glob.glob(os.path.join(DataDir,"trainA/*.{}".format(imgtype))):
                fn2 = fn.replace('trainA','trainB')
                if BtoA:
                    self.dataset.append([[fn2],[fn]])
                else:
                    self.dataset.append([[fn],[fn2]])
        elif datalist == '__test__':
            for fn in glob.glob(os.path.join(DataDir,"testA/*.{}".format(imgtype))):
                fn2 = fn.replace('testA','testB')
                if BtoA:
                    self.dataset.append([[fn2],[fn]])
                else:
                    self.dataset.append([[fn],[fn2]])
        else:
            ## an input/output image can consist of multiple images; they are stacked as channels
            with open(datalist) as input:
                for line in input:
                    files = line.strip().split('\t')
                    if(len(files))<2:
                        self.dataset.append([
                            [os.path.join(DataDir,files[0])],
                            [os.path.join(DataDir,files[0])]
                        ])
                    if(len(files)<len(set(from_col).union(set(to_col)))):
                        print("Error in reading data file: ",files)
                        exit()
                    if BtoA:
                        self.dataset.append([
                            [os.path.join(DataDir,files[i]) for i in to_col],
                            [os.path.join(DataDir,files[i]) for i in from_col]
                        ])
                    else:
                        self.dataset.append([
                            [os.path.join(DataDir,files[i]) for i in from_col],
                            [os.path.join(DataDir,files[i]) for i in to_col]
                        ])
                    for i in set(from_col).union(set(to_col)):
                        if not os.path.isfile(os.path.join(DataDir,files[i])):
                            print("{} not found!".format(os.path.join(DataDir,files[i])))
                            exit()

        self.crop = crop
        self.grey = grey
        self.random = random
        print("Cropped size: ",self.crop, "Clip: ",self.clip)
        print("loaded {} images".format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    def get_img_path(self, i):
        return(self.dataset[i][0][0])

    def var2img(self,var):
        if self.clip[0] is not None and self.clip[1] is not None:            
            return (0.5*(var+1)*(self.clip[1]-self.clip[0])+self.clip[0]).squeeze()
        else:
            return(0.5*(1.0+var)*255).squeeze()
    
    def stack_imgs(self,fns,resize=False, onehot=False):
        imgs_in =[]
        for fn in fns:
            fn1,ext = os.path.splitext(fn)
            # image can be given as csv or jpg/png... etc
            if ext==".csv":
                img_in = np.loadtxt(fn, delimiter=",")
            elif ext==".txt":
                img_in = np.loadtxt(fn)
            elif ext==".npy":
                img_in = np.load(fn)
    #            img_in = (np.sqrt(np.clip(img_in,0,100)))/10.0  ## nasty preprocess
    #            img_in = (img_in - np.mean(img_in))/2*np.std(img_in) # standardize
            elif ext==".dcm":
                ref_dicom_in = dicom.read_file(fn, force=True)
                ref_dicom_in.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
                img_in  = ref_dicom_in.pixel_array+ref_dicom_in.RescaleIntercept
            else:  ## image file
                img_in = read_image(fn, color=not self.grey)/127.5 -1.0

            # make the image shape to [C,H,W]
            if len(img_in.shape) == 2:
                img_in = img_in[np.newaxis,]

            # resize if the image is too small
            if resize:
                if img_in.shape[1]<self.crop[0] or img_in.shape[2]<self.crop[1]:
                    if self.crop[0]/img_in.shape[1] < self.crop[1]/img_in.shape[2]:
                        img_in = resize(img_in, (int(self.crop[1]/img_in.shape[2]*img_in.shape[1]), self.crop[1]))
                    else:
                        img_in = resize(img_in, (self.crop[0], int(self.crop[0]/img_in.shape[1]*img_in.shape[2])))
            imgs_in.append(img_in)

        imgs_in = np.concatenate(imgs_in, axis=0)
    #    print(imgs_in.shape)
        if onehot>0:
            return(np.eye(self.class_num)[imgs_in[0].astype(np.uint64)].astype(np.float32).transpose((2,0,1)))
        else:
            ## clip and normalise to [-1,1]
            if self.clip[0] is not None or self.clip[1] is not None:
                imgs_in = np.clip(imgs_in, self.clip[0], self.clip[1])
            if self.clip[0] is not None and self.clip[1] is not None:            
                imgs_in = 2*(imgs_in-self.clip[0])/(self.clip[1]-self.clip[0]) - 1.0
            return(imgs_in.astype(np.float32))

    def get_example(self, i):
        il,ol = self.dataset[i]
        imgs_in = self.stack_imgs(il)
        imgs_out = self.stack_imgs(ol, onehot=(self.class_num>0) )
#        print(np.min(imgs_in),np.max(imgs_in),np.min(imgs_out),np.max(imgs_out))
        H = self.crop[0] if self.crop[0] else 16*((imgs_in.shape[1]-2*self.random)//16)
        W = self.crop[1] if self.crop[1] else 16*((imgs_in.shape[2]-2*self.random)//16)
        if self.random: # random crop/flip
            if random.choice([True, False]):
                imgs_in = imgs_in[:, :, ::-1]
                imgs_out = imgs_out[:, :, ::-1]
#            if random.choice([True, False]):
#                imgs_in = imgs_in[:, ::-1, :]
#                imgs_out = imgs_out[:, ::-1, :]
            y_offset = random.randint((imgs_in.shape[1]-H)//2-self.random, (imgs_in.shape[1]-H)//2+self.random)
            y_slice = slice(y_offset, y_offset + H)
            x_offset = random.randint((imgs_in.shape[2]-W)//2-self.random, (imgs_in.shape[2]-W)//2+self.random)
            x_slice = slice(x_offset, x_offset + W)
        else: # centre crop
            y_offset = (imgs_in.shape[1] - H) // 2
            x_offset = (imgs_in.shape[2] - W) // 2
            y_slice = slice(y_offset, y_offset + H)
            x_slice = slice(x_offset, x_offset + W)
        return imgs_in[:,y_slice,x_slice], imgs_out[:,y_slice,x_slice]
    
    def overwrite_dicom(self,new,fn,salt):
        ref_dicom = dicom.read_file(fn, force=True)
        dt=ref_dicom.pixel_array.dtype
        img = np.full(ref_dicom.pixel_array.shape, self.clip[0], dtype=np.float32)
        ch,cw = new.shape
        h = self.crop[0] if self.crop[0] else 16*((img.shape[0]-2*self.random)//16)
        w = self.crop[1] if self.crop[1] else 16*((img.shape[1]-2*self.random)//16)
        if np.min(img - ref_dicom.RescaleIntercept)<0:
            ref_dicom.RescaleIntercept = -1024
        img[np.newaxis,(ch-h)//2:(ch+h)//2,(cw-w)//2:(cw+w)//2] = new
        img -= ref_dicom.RescaleIntercept
        img = img.astype(dt)           
        print("min {}, max {}, intercept {}".format(np.min(img),np.max(img),ref_dicom.RescaleIntercept))
#            print(img.shape, img.dtype)
        ref_dicom.PixelData = img.tostring()
        ## UID should be changed for dcm's under different dir
        #                uid=dicom.UID.generate_uid()
        #                uid = dicom.UID.UID(uid.name[:-len(args.suffix)]+args.suffix)
        uid = ref_dicom[0x8,0x18].value.split(".")
        uid[-2] = salt
        uidn = ".".join(uid)
        uid = ".".join(uid[:-1])

#            ref_dicom[0x2,0x3].value=uidn  # Media SOP Instance UID                
        ref_dicom[0x8,0x18].value=uidn  #(0008, 0018) SOP Instance UID              
        ref_dicom[0x20,0xd].value=uid  #(0020, 000d) Study Instance UID       
        ref_dicom[0x20,0xe].value=uid  #(0020, 000e) Series Instance UID
        ref_dicom[0x20,0x52].value=uid  # Frame of Reference UID
        return(ref_dicom)