#!/usr/bin/env python
# -*- coding: utf-8 -*-
# implementation of pix2pix
# By Shizuo Kaji

import os
import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
from chainer.training import extensions
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings
from updater import softmax_focalloss

def var2unit_img(var, base=-1.0, rng=2.0):
    img = var.data.get()
    img = (img - base) / rng  # [0, 1)
    img = img.transpose(0, 2, 3, 1)
    return img

class VisEvaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        params = kwargs.pop('params')
        super(VisEvaluator, self).__init__(*args, **kwargs)
        self.vis_out = params['vis_out']
        self.args= params['args']
        self.count = 0
        warnings.filterwarnings("ignore", category=UserWarning)

    def evaluate(self):
        domain = ['in','truth','out']
        if self.eval_hook:
            self.eval_hook(self)

        for k,dataset in enumerate(['test','train']):
            batch =  self._iterators[dataset].next()
            x_in, t_out = chainer.dataset.concat_examples(batch, self.device)
            x_in = Variable(x_in)   # original image
            t_out = Variable(t_out) # corresponding translated image (ground truth)

            with chainer.using_config('train', False), chainer.function.no_backprop_mode():
                x_out = self._targets['dec_y'](self._targets['enc_x'](x_in)) # translated image by NN

            if k==0:  # for test dataset, compute some statistics
                fig = plt.figure(figsize=(12, 6 * len(batch)))
                gs = gridspec.GridSpec(2* len(batch), 4, wspace=0.1, hspace=0.1)
                loss_rec_L1 = F.mean_absolute_error(x_out, t_out)
                loss_rec_L2 = F.mean_squared_error(x_out, t_out)
                loss_rec_CE = softmax_focalloss(x_out, t_out)
                result = {"myval/loss_L1": loss_rec_L1, "myval/loss_L2": loss_rec_L2, "myval/loss_CE": loss_rec_CE}

            ## iterate over batch
            for i, var in enumerate([x_in, t_out, x_out]):
                if i % 3 != 0 and self.args.class_num>0: # t_out, x_out
                    imgs = var2unit_img(var,0,1) # softmax
                    #imgs[:,:,:,0] = 0 # class 0 => black  ###### TODO
                    #imgs = np.roll(imgs,1,axis=3)[:,:,:,:3]  ## R0B, show only 3 classes (-1,0,1)
                else:
                    imgs = var2unit_img(var) # tanh
#                print(imgs.shape,np.min(imgs),np.max(imgs))
                for j in range(len(imgs)):
                    ax = fig.add_subplot(gs[j+k*len(batch),i])
                    ax.set_title(dataset+"_"+domain[i], fontsize=8)
                    if(imgs[j].shape[2] == 3): ## RGB
                        ax.imshow(imgs[j], interpolation='none',vmin=0,vmax=1)
                    elif(imgs[j].shape[2] >= 4): ## categorical
                        cols = ['k','r','g','b','c','m','y']*5
                        cmap = colors.ListedColormap(cols)
                        im = np.argmax(imgs[j], axis=2)
                        norm = colors.BoundaryNorm(list(range(len(cols)+1)), cmap.N)
                        ax.imshow(im, interpolation='none', cmap=cmap, norm=norm)
                    else:
                        ax.imshow(imgs[j][:,:,-1], interpolation='none',cmap='gray',vmin=0,vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])

            ## difference image
            diff = (x_out-t_out).data.get().transpose(0, 2, 3, 1)
            for j in range(len(diff)):
                ax = fig.add_subplot(gs[j+k*len(batch),3])
                ax.imshow(diff[j][:,:,0], interpolation='none',cmap='coolwarm',vmin=-0.1,vmax=0.1)

        gs.tight_layout(fig)
        plt.savefig(os.path.join(self.vis_out,'count{:0>4}.jpg'.format(self.count)), dpi=200)
        self.count += 1
        plt.close()

        return result
