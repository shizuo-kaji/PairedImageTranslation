#!/usr/bin/env python
#############################
##
## Image converter by learned models
##
#############################

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import glob
import json
import codecs
from datetime import datetime as dt
import time
import chainer.cuda
from chainer import serializers, Variable
import numpy as np
import net
import random
import chainer.functions as F
from chainercv.utils import write_image
from chainercv.transforms import resize
from chainerui.utils import save_args
from arguments import arguments 
from consts import activation,dtypes


if __name__ == '__main__':
    args = arguments()
    args.suffix = "out"
    outdir = os.path.join(args.out, dt.now().strftime('out_%m%d_%H%M'))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        print('use gpu {}'.format(args.gpu))

    ## load arguments from "arg" file used in training
    if args.argfile:
        with open(args.argfile, 'r') as f:
            larg = json.load(f)
            root=os.path.dirname(args.argfile)
            for x in ['grey',
              'dis_norm','dis_activation','dis_basech','dis_ksize','dis_sample','dis_down','dis_ndown',
              'gen_norm','gen_activation','gen_out_activation','gen_nblock','gen_chs','gen_sample','gen_down','gen_up','gen_ksize','unet',
              'gen_fc','gen_fc_activation','spconv','eqconv','dtype']:
                if x in larg:
                    setattr(args, x, larg[x])
            if not args.model_gen:
                if larg["epoch"]:
                    args.load_models=os.path.join(root,'gen_g{}.npz'.format(larg["epoch"]))
                else:
                    args.load_models=os.path.join(root,'gen_g{}.npz'.format(larg["lrdecay_start"]+larg["lrdecay_period"]))
                    
    args.random = False
    save_args(args, outdir)
    args.dtype = dtypes[args.dtype]
    args.dis_activation = activation[args.dis_activation]
    args.gen_activation = activation[args.gen_activation]
    args.gen_out_activation = activation[args.gen_out_activation]
    args.gen_fc_activation = activation[args.gen_fc_activation]
    print(args)
    chainer.config.dtype = args.dtype

    ## load images
    if args.imgtype=="dcm":
        from dataset_dicom import Dataset
    else:
        from dataset import Dataset  
    if args.val:
        dataset = Dataset(args.val, args.root, args.from_col, args.from_col, crop=(args.crop_height,args.crop_width), random=False, grey=args.grey)
    elif args.train:
        dataset = Dataset(args.train, args.root, args.from_col, args.from_col, crop=(args.crop_height,args.crop_width), random=False, grey=args.grey)
    else:
        print("Load Dataset from disk: {}".format(args.root))
        with open(os.path.join(args.out,"filenames.txt"),'w') as output:
            for file in glob.glob(os.path.join(args.root,"**/*.{}".format(args.imgtype)), recursive=True):
                output.write('{}\n'.format(file))
        dataset = Dataset(os.path.join(args.out,"filenames.txt"), "", [0], [0], crop=(args.crop_height,args.crop_width), random=False, grey=args.grey)
        
#    iterator = chainer.iterators.MultiprocessIterator(dataset, args.batch_size, n_processes=3, repeat=False, shuffle=False)
    iterator = chainer.iterators.MultithreadIterator(dataset, args.batch_size, n_threads=3, repeat=False, shuffle=False)   ## best performance
#    iterator = chainer.iterators.SerialIterator(dataset, args.batch_size,repeat=False, shuffle=False)

    args.ch = len(dataset[0][0])
    args.out_ch = len(dataset[0][1])
    print("Input channels {}, Output channels {}".format(args.ch,args.out_ch))

    ## load generator models
    if args.model_gen:
            gen = net.Generator(args)
            print('Loading {:s}..'.format(args.model_gen))
            serializers.load_npz(args.model_gen, gen)
            if args.gpu >= 0:
                gen.to_gpu()
            xp = gen.xp
    else:
        print("Specify a learnt model.")
        exit()        

    ## start measuring timing
    os.makedirs(outdir, exist_ok=True)
    start = time.time()

    cnt = 0
    salt = str(random.randint(1000, 999999))
    for batch in iterator:
        x_in, t_out = chainer.dataset.concat_examples(batch, device=args.gpu)
        imgs = Variable(x_in)
        with chainer.using_config('train', False), chainer.function.no_backprop_mode():
            out_v = gen(imgs)
        if args.gpu >= 0:
            imgs = xp.asnumpy(imgs.data)
            out = xp.asnumpy(out_v.data)
        else:
            imgs = imgs.data
            out = out_v.data
        
        ## output images
        for i in range(len(out)):
            fn = dataset.get_img_path(cnt)
            print("\nProcessing {}".format(fn))
            new = dataset.var2img(out[i]) 
            print("raw value: {} {}".format(np.min(out[i]),np.max(out[i])))
            path = os.path.join(outdir,os.path.basename(fn))
            # converted image
            if args.imgtype=="dcm":
                ref_dicom = dataset.overwrite(new[0],fn,salt)
                ref_dicom.save_as(path)
            else:
                write_image(new, path)

            cnt += 1
        ####

    elapsed_time = time.time() - start
    print ("{} images in {} sec".format(cnt,elapsed_time))



