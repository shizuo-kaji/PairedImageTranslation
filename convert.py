#!/usr/bin/env python
#############################
##
## Image converter by learned models
##
#############################

import argparse
import os,glob
import json,codecs
from datetime import datetime as dt
import time
import numpy as np
import net
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers, Variable, cuda
from chainercv.utils import write_image
from chainercv.transforms import resize
from chainerui.utils import save_args
from arguments import arguments 
from consts import dtypes
from dataset import Dataset  

#os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    args = arguments()
    args.random = 0 ## necessary to infer crop size
    outdir = os.path.join(args.out, dt.now().strftime('out_%Y%m%d_%H%M'))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        print('use gpu {}'.format(args.gpu))

    # infer model names
    if not args.model_gen:
        root=os.path.dirname(args.argfile)
        args.model_gen=os.path.join(root,'enc_x{}.npz'.format(args.epoch))
        if not os.path.isfile(args.model_gen):
            args.model_gen = args.model_gen.replace('enc_x','gen_')

    save_args(args, outdir)
    print(args)
    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]

    ## load images
    if os.path.isfile(args.val):
        dataset = Dataset(args.val, args.root, args.from_col, args.from_col, clipA=args.clipA, clipB=args.clipB, crop=(args.crop_height,args.crop_width), imgtype=args.imgtype, class_num=args.class_num, stack=args.stack, grey=args.grey, BtoA=args.btoa)
    else:
        print("Load Dataset from directory: {}".format(args.root))
        dataset = Dataset('__convert__', args.root, [0], [0], clipA=args.clipA, clipB=args.clipB, crop=(args.crop_height,args.crop_width), imgtype=args.imgtype, class_num=args.class_num, stack=args.stack, grey=args.grey, BtoA=args.btoa, fn_pattern=args.fn_pattern)
        
    #iterator = chainer.iterators.MultiprocessIterator(dataset, args.batch_size, n_processes=4, repeat=False, shuffle=False)
    iterator = chainer.iterators.MultithreadIterator(dataset, args.batch_size, n_threads=3, repeat=False, shuffle=False)
#    iterator = chainer.iterators.SerialIterator(dataset, args.batch_size,repeat=False, shuffle=False)

    if args.ch != len(dataset[0][0]):
        print("number of input channels is different during training.")
    print("Input channels {}, Output channels {}".format(args.ch,args.out_ch))

    ## load generator models
    if "enc" in args.model_gen:
        if (args.gen_pretrained_encoder and args.gen_pretrained_lr_ratio == 0):
            if "resnet" in args.gen_pretrained_encoder:
                pretrained = L.ResNet50Layers()
                print("Pretrained ResNet model loaded.")
            else:
                pretrained = L.VGG16Layers()
                print("Pretrained VGG model loaded.")
            if args.gpu >= 0:
                pretrained.to_gpu()
            enc = net.Encoder(args, pretrained)
        else:
            enc = net.Encoder(args)
        print('Loading {:s}..'.format(args.model_gen))
        serializers.load_npz(args.model_gen, enc)
        dec = net.Decoder(args)
        modelfn = args.model_gen.replace('enc_x','dec_y')
        modelfn = modelfn.replace('enc_y','dec_x')
        print('Loading {:s}..'.format(modelfn))
        serializers.load_npz(modelfn, dec)
        if args.gpu >= 0:
            enc.to_gpu()
            dec.to_gpu()
        xp = enc.xp
        is_AE = True
    elif "gen" in args.model_gen:
        gen = net.Generator(args)
        print('Loading {:s}..'.format(args.model_gen))
        serializers.load_npz(args.model_gen, gen)
        if args.gpu >= 0:
            gen.to_gpu()
        xp = gen.xp
        is_AE = False
    elif "identity" == args.model_gen:
        gen = F.identity
        print("Identity..")
        xp = np
        is_AE = False
    else:
        print("Specify a learned model.")
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
            if is_AE:
                x_out = dec(enc(imgs))
            else:
                x_out = gen(imgs)
        ## unfold stack and apply softmax
        if args.stack>0:
            x_out = x_out.reshape(x_out.shape[0]*args.stack,x_out.shape[1]//args.stack,x_out.shape[2],x_out.shape[3])
        if args.class_num>0:
            x_out = F.softmax(x_out)
        if args.gpu >= 0:
            imgs.to_cpu()
            x_out.to_cpu() 
        imgs = imgs.data
        out = x_out.data[args.stack//2::args.stack]        ## use the only middle slice in the stack

        ## output images
        for i in range(len(out)):
            fn = dataset.get_img_path(cnt)
            bfn,ext = os.path.splitext(fn)
            bfn = os.path.basename(bfn)
            relfn = os.path.relpath(fn,args.root)
            os.makedirs(os.path.join(outdir, os.path.dirname(relfn)), exist_ok=True)
            print("Processing {}".format(fn))
            if args.class_num>0:  ## TODO: stacked
                #write_image((255*np.stack([out[i,2],np.zeros_like(out[i,0]),out[i,1]],axis=0)).astype(np.uint8), os.path.join(outdir,bfn)+".jpg")
                new = np.argmax(out[i],axis=0)
                airvalue = 0
#                print(new.shape)
            else:
                airvalue = None
                new = dataset.var2img(out[i],args.clipB)
            if args.vis_freq>0 and cnt%args.vis_freq==0:
                print("raw value: {} -- {}".format(np.min(out[i]),np.max(out[i])))
                print("image value: {} -- {}, ".format(np.min(new),np.max(new), new.shape))
            # converted image
            if args.imgtype=="dcm":
                path = os.path.join(outdir,relfn) ## preserve directory structures
                print(path)
                ref_dicom = dataset.overwrite_dicom(new,fn,salt,airvalue=airvalue)
                ref_dicom.save_as(path)
            elif args.imgtype=="npy":
                path = os.path.join(outdir,bfn)
                np.save(path,new)
            elif args.imgtype=="txt":
                path = os.path.join(outdir,bfn)+".txt"
                np.savetxt(path,new,fmt="%d")
            else:
            # save image
                path = os.path.join(outdir,bfn)+".jpg"
                write_image(new, path)

            cnt += 1
        ####

    elapsed_time = time.time() - start
    print ("\n{} images in {} sec".format(cnt,elapsed_time))
    print ("Output in {}".format(outdir))
    iterator.finalize()
    exit()



