#!/usr/bin/env python
# -*- coding: utf-8 -*-
# implementation of pix2pix
# By Shizuo Kaji

from __future__ import print_function
import argparse
import os
from datetime import datetime as dt
import numpy as np

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer import training,serializers
from chainer.training import extensions
#from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
import chainer.functions as F
from chainer.dataset import convert

import net
from updater import pixupdater
from arguments import arguments 

from dataset import Dataset
from visualizer import VisEvaluator
from consts import dtypes,optim

def main():
    args = arguments()

#    chainer.config.type_check = False
    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]
    chainer.print_runtime_info()
    #print('Chainer version: ', chainer.__version__)
    #print('GPU availability:', chainer.cuda.available)
    #print('cuDNN availability:', chainer.cuda.cudnn_enabled)

    ## dataset preparation
    if args.imgtype=="dcm":
        from dataset_dicom import Dataset
    else:
        from dataset import Dataset  
    train_d = Dataset(args.train, args.root, args.from_col, args.to_col, crop=(args.crop_height,args.crop_width), imgtype=args.imgtype, random=args.random_translate, grey=args.grey, BtoA=args.btoa)
    test_d = Dataset(args.val, args.root, args.from_col, args.to_col, crop=(args.crop_height,args.crop_width), imgtype=args.imgtype, random=args.random_translate, grey=args.grey, BtoA=args.btoa)
    args.crop_height,args.crop_width = train_d.crop

    # setup training/validation data iterators
    train_iter = chainer.iterators.SerialIterator(train_d, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test_d, args.nvis, shuffle=False)
    test_iter_gt = chainer.iterators.SerialIterator(train_d, args.nvis, shuffle=False)   ## same as training data; used for validation

    args.ch = len(train_d[0][0])
    args.out_ch = len(train_d[0][1])
    print("Input channels {}, Output channels {}".format(args.ch,args.out_ch))

    ## Set up models
    gen = net.Generator(args)
    dis = net.Discriminator(args)

    ## load learnt models
    optimiser_files = []
    if args.model_gen:
        serializers.load_npz(args.model_gen, gen)
        print('model loaded: {}'.format(args.model_gen))
        optimiser_files.append(args.model_gen.replace('gen_','opt_gen_'))
    if args.model_dis:
        serializers.load_npz(args.model_dis, dis)
        print('model loaded: {}'.format(args.model_dis))
        optimiser_files.append(args.model_dis.replace('dis_','opt_dis_'))

    ## send models to GPU
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # Setup optimisers
    def make_optimizer(model, lr, opttype='Adam'):
#        eps = 1e-5 if args.dtype==np.float16 else 1e-8
        optimizer = optim[opttype](lr)
        if args.weight_decay>0:
            if opttype in ['Adam','AdaBound','Eve']:
                optimizer.weight_decay_rate = args.weight_decay
            else:
                if args.weight_decay_norm =='l2':
                    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
                else:
                    optimizer.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))
        optimizer.setup(model)
        return optimizer

    opt_gen = make_optimizer(gen,args.learning_rate_gen,args.optimizer)
    opt_dis = make_optimizer(dis,args.learning_rate_dis,args.optimizer)
    optimizers = {'opt_g':opt_gen, 'opt_d':opt_dis}

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

    # Set up trainer
    updater = pixupdater(
        models=(gen, dis),
        iterator={
            'main': train_iter},
        optimizer={
            'gen': opt_gen,
            'dis': opt_dis},
#        converter=convert.ConcatWithAsyncTransfer(),
        params={'args': args},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    ## save learnt results at an interval
    if args.snapinterval<0:
        args.snapinterval = args.epoch
    snapshot_interval = (args.snapinterval, 'epoch')
    display_interval = (args.display_interval, 'iteration')
        
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt_gen, 'opt_gen_{.updater.epoch}.npz'), trigger=snapshot_interval)
    if args.lambda_dis>0:
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_{.updater.epoch}.npz'), trigger=snapshot_interval)
        trainer.extend(extensions.dump_graph('dis/loss_real', out_name='dis.dot'))
        trainer.extend(extensions.snapshot_object(
            opt_dis, 'opt_dis_{.updater.epoch}.npz'), trigger=snapshot_interval)

    if args.lambda_rec_l1 > 0:
        trainer.extend(extensions.dump_graph('gen/loss_L1', out_name='gen.dot'))
    elif args.lambda_rec_l2 > 0:
        trainer.extend(extensions.dump_graph('gen/loss_L2', out_name='gen.dot'))

    ## log outputs
    log_keys = ['epoch', 'iteration','lr']
    log_keys_gen = ['gen/loss_L1', 'gen/loss_L2', 'gen/loss_dis', 'myval/loss_L2', 'gen/loss_tv']
    log_keys_dis = ['dis/loss_real','dis/loss_fake','dis/loss_mispair']
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(log_keys+log_keys_gen+log_keys_dis), trigger=display_interval)
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(log_keys_gen, 'iteration', trigger=display_interval, file_name='loss_gen.png'))
        trainer.extend(extensions.PlotReport(log_keys_dis, 'iteration', trigger=display_interval, file_name='loss_dis.png'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
#    trainer.extend(extensions.ParameterStatistics(gen))
    # learning rate scheduling
    if args.optimizer in ['SGD','Momentum','AdaGrad','RMSprop']:
        trainer.extend(extensions.observe_lr(optimizer_name='gen'), trigger=display_interval)
        trainer.extend(extensions.ExponentialShift('lr', 0.33, optimizer=opt_gen), trigger=(args.epoch/5, 'epoch'))
        trainer.extend(extensions.ExponentialShift('lr', 0.33, optimizer=opt_dis), trigger=(args.epoch/5, 'epoch'))
    elif args.optimizer in ['Adam','AdaBound','Eve']:
        trainer.extend(extensions.observe_lr(optimizer_name='gen'), trigger=display_interval)
        trainer.extend(extensions.ExponentialShift("alpha", 0.33, optimizer=opt_gen), trigger=(args.epoch/5, 'epoch'))
        trainer.extend(extensions.ExponentialShift("alpha", 0.33, optimizer=opt_dis), trigger=(args.epoch/5, 'epoch'))

    # evaluation
    vis_folder = os.path.join(args.out, "vis")
    os.makedirs(vis_folder, exist_ok=True)
    if not args.vis_freq:
        args.vis_freq = len(train_d)//2        
    trainer.extend(VisEvaluator({"test":test_iter, "train":test_iter_gt}, {"gen":gen},
            params={'vis_out': vis_folder}, device=args.gpu),trigger=(args.vis_freq, 'iteration') )

    # ChainerUI: removed until ChainerUI updates to be compatible with Chainer 6.0
#    trainer.extend(CommandsExtension())

    # Run the training
    print("trainer start")
    trainer.run()

if __name__ == '__main__':
    main()
