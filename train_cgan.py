#!/usr/bin/env python
# -*- coding: utf-8 -*-
# implementation of pix2pix
# By Shizuo Kaji

from __future__ import print_function
import os,sys
from datetime import datetime as dt
import numpy as np

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer import training,serializers
from chainer.training import extensions
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
import chainer.functions as F
import chainer.links as L
from chainer.dataset import convert

import net
from updater import Updater
from arguments import arguments 
from dataset import Dataset
from visualizer import VisEvaluator
from consts import dtypes,optim
from dataset import Dataset  
from cosshift import CosineShift

def plot_log(f,a,summary):
    a.set_yscale('log')

def main():
    args = arguments()
    outdir = os.path.join(args.out, dt.now().strftime('%Y%m%d_%H%M')+"_cgan")

#    chainer.config.type_check = False
    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]
    chainer.print_runtime_info()
    #print('Chainer version: ', chainer.__version__)
    #print('GPU availability:', chainer.cuda.available)
    #print('cuDNN availability:', chainer.cuda.cudnn_enabled)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    ## dataset preparation
    train_d = Dataset(args.train, args.root, args.from_col, args.to_col, clipA=args.clipA, clipB=args.clipB, class_num=args.class_num, crop=(args.crop_height,args.crop_width), imgtype=args.imgtype, random_tr=args.random_translate, random_rot=args.random_rotation, random_scale=args.random_scale, stack=args.stack, grey=args.grey, BtoA=args.btoa, fn_pattern=args.fn_pattern)
    test_d = Dataset(args.val, args.root, args.from_col, args.to_col, clipA=args.clipA, clipB=args.clipB, class_num=args.class_num, crop=(args.crop_height,args.crop_width), imgtype=args.imgtype, stack=args.stack, grey=args.grey, BtoA=args.btoa, fn_pattern=args.fn_pattern)
    args.crop_height,args.crop_width = train_d.crop
    if(len(train_d)==0):
        print("No images found!")
        exit()

    # setup training/validation data iterators
    train_iter = chainer.iterators.SerialIterator(train_d, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_d, args.nvis, shuffle=args.vis_random)
    test_iter_gt = chainer.iterators.SerialIterator(train_d, args.nvis, shuffle=args.vis_random)   ## same as training data; used for validation

    args.ch = len(train_d[0][0])
    args.out_ch = len(train_d[0][1])
    print("Input channels {}, Output channels {}".format(args.ch,args.out_ch))
    if(len(train_d)*len(test_d)==0):
        print("No images found!")
        exit()

    ## Set up models
    # shared pretrained layer
    if (args.gen_pretrained_encoder and args.gen_pretrained_lr_ratio == 0):
        if "resnet" in args.gen_pretrained_encoder:
            pretrained = L.ResNet50Layers()
            print("Pretrained ResNet model loaded.")
        else:
            pretrained = L.VGG16Layers()
            print("Pretrained VGG model loaded.")
        if args.gpu >= 0:
            pretrained.to_gpu()
        enc_x = net.Encoder(args, pretrained)
    else:
        enc_x = net.Encoder(args)

#    gen = net.Generator(args)
    dec_y = net.Decoder(args)

    if args.lambda_dis>0:
        dis = net.Discriminator(args)
        models = {'enc_x': enc_x, 'dec_y': dec_y, 'dis': dis}
    else:
        dis = L.Linear(1,1)
        models = {'enc_x': enc_x, 'dec_y': dec_y}


    ## load learnt models
    optimiser_files = []
    if args.model_gen:
        serializers.load_npz(args.model_gen, enc_x)
        serializers.load_npz(args.model_gen.replace('enc_x','dec_y'), dec_y)
        print('model loaded: {}, {}'.format(args.model_gen, args.model_gen.replace('enc_x','dec_y')))
        optimiser_files.append(args.model_gen.replace('enc_x','opt_enc_x'))
        optimiser_files.append(args.model_gen.replace('enc_x','opt_dec_y'))
    if args.model_dis:
        serializers.load_npz(args.model_dis, dis)
        print('model loaded: {}'.format(args.model_dis))
        optimiser_files.append(args.model_dis.replace('dis','opt_dis'))

    ## send models to GPU
    if args.gpu >= 0:
        enc_x.to_gpu()
        dec_y.to_gpu()
        dis.to_gpu()

    # Setup optimisers
    def make_optimizer(model, lr, opttype='Adam', pretrained_lr_ratio=1.0):
#        eps = 1e-5 if args.dtype==np.float16 else 1e-8
        optimizer = optim[opttype](lr)
        optimizer.setup(model)
        if args.weight_decay>0:
            if opttype in ['Adam','AdaBound','Eve']:
                optimizer.weight_decay_rate = args.weight_decay
            else:
                if args.weight_decay_norm =='l2':
                    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
                else:
                    optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))
        return optimizer

    opt_enc_x = make_optimizer(enc_x,args.learning_rate_gen,args.optimizer)
    opt_dec_y = make_optimizer(dec_y,args.learning_rate_gen,args.optimizer)
    opt_dis = make_optimizer(dis,args.learning_rate_dis,args.optimizer)

    optimizers = {'enc_x':opt_enc_x, 'dec_y':opt_dec_y, 'dis':opt_dis}

    ## resume optimisers from file
    if args.load_optimizer:
        for (m,e) in zip(optimiser_files,optimizers):
            if m:
                try:
                    serializers.load_npz(m, optimizers[e])
                    print('optimiser loaded: {}'.format(m))
                except:
                    print("couldn't load {}".format(m))
                    pass

    # finetuning
    if args.gen_pretrained_encoder:
        if args.gen_pretrained_lr_ratio == 0:
            enc_x.base.disable_update()
        else:
            for func_name in enc_x.encoder.base._children:
                for param in enc_x.encoder.base[func_name].params():
                    param.update_rule.hyperparam.eta *= args.gen_pretrained_lr_ratio

    # Set up trainer
    updater = Updater(
        models=(enc_x, dec_y, dis),
        iterator={
            'main': train_iter},
        optimizer=optimizers,
#        converter=convert.ConcatWithAsyncTransfer(),
        params={'args': args},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    ## save learnt results at a specified interval or at the end of training
    if args.snapinterval<0:
        args.snapinterval = args.epoch
    snapshot_interval = (args.snapinterval, 'epoch')
    display_interval = (args.display_interval, 'iteration')
        
    for e in models:
        trainer.extend(extensions.snapshot_object(
            models[e], e+'{.updater.epoch}.npz'), trigger=snapshot_interval)
        if args.parameter_statistics:
            trainer.extend(extensions.ParameterStatistics(models[e]))   ## very slow
    for e in optimizers:
        trainer.extend(extensions.snapshot_object(
            optimizers[e], 'opt_'+e+'{.updater.epoch}.npz'), trigger=snapshot_interval)

    ## plot NN graph
    if args.lambda_rec_l1 > 0:
        trainer.extend(extensions.dump_graph('dec_y/loss_L1', out_name='enc.dot'))
    elif args.lambda_rec_l2 > 0:
        trainer.extend(extensions.dump_graph('dec_y/loss_L2', out_name='gen.dot'))
    elif args.lambda_rec_ce > 0:
        trainer.extend(extensions.dump_graph('dec_y/loss_CE', out_name='gen.dot'))
    if args.lambda_dis>0 and args.dis_warmup < 0:
        trainer.extend(extensions.dump_graph('dis/loss_real', out_name='dis.dot'))

    ## log outputs
    log_keys = ['epoch', 'iteration','lr']
    log_keys_gen = ['myval/loss_L1', 'myval/loss_L2']
    log_keys_dis = []
    if args.lambda_rec_l1 > 0:
        log_keys_gen.append('dec_y/loss_L1')
    if args.lambda_rec_l2 > 0:
        log_keys_gen.append('dec_y/loss_L2')
    if args.lambda_dice > 0:
        log_keys_gen.append('dec_y/loss_dice')
    if args.lambda_rec_ce > 0:
        log_keys_gen.extend(['dec_y/loss_CE','myval/loss_CE'])
    if args.lambda_reg>0:
        log_keys.extend(['enc_x/loss_reg'])
    if args.lambda_tv > 0:
        log_keys_gen.append('dec_y/loss_tv')     
    if args.lambda_dis>0:
        log_keys_dis.extend(['dec_y/loss_dis','dis/loss_real','dis/loss_fake'])
    if args.lambda_mispair > 0:
        log_keys_dis.append('dis/loss_mispair')     
    if args.dis_wgan:
        log_keys_dis.extend(['dis/loss_gp'])
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(log_keys+log_keys_gen+log_keys_dis), trigger=display_interval)
    if extensions.PlotReport.available():
#        trainer.extend(extensions.PlotReport(['lr'], 'iteration',trigger=display_interval, file_name='lr.png'))
        trainer.extend(extensions.PlotReport(log_keys_gen, 'iteration', trigger=display_interval, file_name='loss_gen.png', postprocess=plot_log))
        trainer.extend(extensions.PlotReport(log_keys_dis, 'iteration', trigger=display_interval, file_name='loss_dis.png'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # learning rate scheduling
    trainer.extend(extensions.observe_lr(optimizer_name='enc_x'), trigger=display_interval)
    if args.optimizer in ['Adam','AdaBound','Eve']:
        lr_target = 'eta'
    else:
        lr_target = 'lr'
    if args.lr_drop > 0:  ## cosine annealing
        for e in [opt_enc_x,opt_dec_y,opt_dis]:
            trainer.extend(CosineShift(lr_target, args.epoch//args.lr_drop, optimizer=e), trigger=(1, 'epoch'))
            #trainer.extend(extensions.ExponentialShift(lr_target, 0.33, optimizer=e), trigger=(args.epoch//args.lr_drop, 'epoch'))
    else:
        decay_end_iter = args.epoch*len(train_d)
        for e in [opt_enc_x,opt_dec_y,opt_dis]:
            trainer.extend(extensions.LinearShift(lr_target, (1.0,0.0), (decay_end_iter//2,decay_end_iter), optimizer=e))

    # evaluation
    vis_folder = os.path.join(outdir, "vis")
    os.makedirs(vis_folder, exist_ok=True)
    if not args.vis_freq:
        args.vis_freq = max(len(train_d)//2,50)        
    trainer.extend(VisEvaluator({"test":test_iter, "train":test_iter_gt}, {"enc_x":enc_x, "dec_y": dec_y},
            params={'vis_out': vis_folder, 'args': args}, device=args.gpu),trigger=(args.vis_freq, 'iteration') )

    # ChainerUI: removed until ChainerUI updates to be compatible with Chainer 6.0
    trainer.extend(CommandsExtension())

    # Run the training
    print("\nresults are saved under: ",outdir)
    save_args(args, outdir)
    with open(os.path.join(outdir,"args.txt"), 'w') as fh:
        fh.write(" ".join(sys.argv))
    trainer.run()

if __name__ == '__main__':
    main()
