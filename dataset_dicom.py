import os
import pydicom as dicom
import random
import glob

from chainer.dataset import dataset_mixin
import numpy as np
#from skimage.transform import rescale
from chainercv.transforms import random_crop,center_crop, resize
from consts import dtypes

class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, datalist, DataDir, from_col, to_col, crop=(None,None), random=False, grey=True, imgtype='dcm', BtoA=False):
        self.dataset = []
        self.base = -1024
        self.range = 2000
        if not crop[0]:
            self.crop = (384,480)  ## default for the CBCT dataset
        else:
            self.crop = crop
        self.grey = True
        self.random = random # random crop/flip for data augmentation
        self.dtype = np.float32
        ## an input/output image can consist of multiple images; they are stacked as channels
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
            with open(datalist) as input:
                for line in input:
                    files = line.strip().split('\t')
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
                    for i in range(len(files)):
                        if not os.path.isfile(os.path.join(DataDir,files[i])):
                            print("{} not found!".format(os.path.join(DataDir,files[i])))
                            exit()
                            
        print("loaded {} images".format(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    def get_img_path(self, i):
        return '{:s}'.format(self.dataset[i][0][0])

    def get_example(self, i):
        il,ol = self.dataset[i]
        ref_dicom_in = dicom.read_file(il[0], force=True)
        ref_dicom_in.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
        imgs_in  = self.img2var(ref_dicom_in.pixel_array.astype(self.dtype) +ref_dicom_in.RescaleIntercept)[np.newaxis,:,:]
        if(il[0]==ol[0]):
            imgs_out = imgs_in.copy()
        else:
            ref_dicom_out = dicom.read_file(ol[0], force=True)
            ref_dicom_out.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
            imgs_out = self.img2var(ref_dicom_out.pixel_array.astype(self.dtype)+ref_dicom_out.RescaleIntercept)[np.newaxis,:,:]
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
            y_offset = int(round((imgs_in.shape[1]-H) / 2.))
            x_offset = int(round((imgs_in.shape[2]-W) / 2.))
            y_slice = slice(y_offset, y_offset + H)
            x_slice = slice(x_offset, x_offset + W)
        return imgs_in[:,y_slice,x_slice], imgs_out[:,y_slice,x_slice]


    def img2var(self,img):
        # cut off mask [-1,1] or [0,1] output
        return(2*(np.clip(img,self.base,self.base+self.range)-self.base)/self.range-1.0)
#        return((np.clip(img,self.base,self.base+self.range)-self.base)/self.range)
    
    def var2img(self,var):
        return(0.5*(1.0+var)*self.range + self.base)
#        return(np.round(var*self.range + self.base))

    def overwrite(self,new,fn,salt):
        ref_dicom = dicom.read_file(fn, force=True)
        dt=ref_dicom.pixel_array.dtype
        img = np.full(ref_dicom.pixel_array.shape, self.base, dtype=np.float32)
        ch,cw = new.shape
        h,w = self.crop
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
