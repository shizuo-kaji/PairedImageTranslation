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
import warnings

def postprocess(var):
    img = var.data.get()
    img = (img + 1.0) / 2.0  # [0, 1)
    img = img.transpose(0, 2, 3, 1)
    return img

def softmax_focalloss(x, t, class_num=4, gamma=2, eps=1e-7):
    p = F.clip(F.softmax(x), x_min=eps, x_max=1-eps)
    q = -t * F.log(p)
    return F.sum(q * ((1 - p) ** gamma))

class VisEvaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        params = kwargs.pop('params')
        super(VisEvaluator, self).__init__(*args, **kwargs)
        self.vis_out = params['vis_out']
        self.count = 0
        warnings.filterwarnings("ignore", category=UserWarning)

    def evaluate(self):
        if self.eval_hook:
            self.eval_hook(self)

        for k,dataset in enumerate(['test','train']):
            batch =  self._iterators[dataset].next()
            x_in, t_out = chainer.dataset.concat_examples(batch, self.device)
            x_in = Variable(x_in)
            t_out = Variable(t_out)

            with chainer.using_config('train', False), chainer.function.no_backprop_mode():
                x_out = self._targets['gen'](x_in)
            
            if k==0:
                fig = plt.figure(figsize=(9, 6 * len(batch)))
                gs = gridspec.GridSpec(2* len(batch), 3, wspace=0.1, hspace=0.1)
                loss_rec_L1 = F.mean_absolute_error(x_out, t_out)
                loss_rec_L2 = F.mean_squared_error(x_out, t_out)
                loss_rec_CE = softmax_focalloss(x_out, t_out)
                result = {"myval/loss_L1": loss_rec_L1, "myval/loss_L2": loss_rec_L2, "myval/loss_CE": loss_rec_CE}

            if x_out.shape[1]>3:
                x_out = F.softmax(x_out)

            for i, var in enumerate([x_in, t_out, x_out]):
                imgs = postprocess(var)
                for j in range(len(imgs)):
                    ax = fig.add_subplot(gs[j+k*len(batch),i])
                    if(imgs[j].shape[2] == 3):
                        ax.imshow(imgs[j], interpolation='none',vmin=0,vmax=1)
                    elif(imgs[j].shape[2] == 4):
                        ax.imshow(imgs[j][:,:,1:], interpolation='none',vmin=0,vmax=1)
                    else:
                        ax.imshow(imgs[j][:,:,-1], interpolation='none',cmap='gray',vmin=0,vmax=1)
                    ax.set_xticks([])
                    ax.set_yticks([])

        gs.tight_layout(fig)
        plt.savefig(os.path.join(self.vis_out,'count{:0>4}.jpg'.format(self.count)), dpi=200)
        self.count += 1
        plt.close()

        return result
