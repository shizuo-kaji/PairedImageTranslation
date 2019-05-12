from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable,cuda

import random
import numpy as np
from PIL import Image

from chainer import function
from chainer.utils import type_check

def add_noise(h, sigma): 
    if chainer.config.train and sigma>0:
        return h + sigma * h.xp.random.randn(*h.shape, dtype=h.dtype)
    else:
        return h

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        xp = cuda.get_array_module(images)
        for image in images:
            image = xp.expand_dims(image, axis=0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = xp.copy(self.images[random_id])
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = xp.concatenate(return_images)
        return return_images

class pixupdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        params = kwargs.pop('params')
        super(pixupdater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.init_alpha = self.get_optimizer('gen').alpha
        self.xp = self.gen.xp
        self._buffer = ImagePool(50 * self.args.batch_size)

    def loss_func_comp(self, y, val, noise=0):
        if noise>0:
            val += random.normalvariate(0,noise)   ## jitter for the target value
        target = self.xp.full(y.data.shape, val, dtype=y.dtype)
        return F.mean_squared_error(y, target)

    def total_variation(self,x,tau=1e-6):
        xp = cuda.get_array_module(x.data)
        wh = xp.tile(xp.asarray([[[[1,0],[-1,0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
        ww = xp.tile(xp.asarray([[[[1, -1],[0, 0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
        dx = F.convolution_2d(x, W=wh)
        dy = F.convolution_2d(x, W=ww)
        d = F.sqrt(dx**2 + dy**2 + xp.full(dx.data.shape, tau**2, dtype=dx.dtype))
        return(F.average(d))

    def update_core(self):        
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis

        ## decay learning rate
        if self.is_new_epoch and self.epoch >= self.args.lrdecay_start:
            decay_step = self.init_alpha / self.args.lrdecay_period
#            print('lr decay', decay_step)
            if gen_optimizer.alpha > decay_step:
                gen_optimizer.alpha -= decay_step
            if dis_optimizer.alpha > decay_step:
                dis_optimizer.alpha -= decay_step

        ## image conversion
        batch = self.get_iterator('main').next()
        x_in, t_out = self.converter(batch, self.device)
        x_out = gen(add_noise(Variable(x_in), sigma=self.args.noise))
        x_in_out_copy = Variable(self._buffer.query(F.concat([x_in,x_out]).data))

        # reconstruction error
        loss_rec_l1 = F.mean_absolute_error(x_out, t_out)
        loss_rec_l2 = F.mean_squared_error(x_out, t_out)
        chainer.report({'loss_L1': loss_rec_l1}, gen)
        chainer.report({'loss_L2': loss_rec_l2}, gen)
        loss_gen = self.args.lambda_rec_l1*loss_rec_l1 + self.args.lambda_rec_l2*loss_rec_l2

        if self.args.lambda_tv > 0:
            loss_tv = self.total_variation(x_out, self.args.tv_tau)
            loss_gen += self.args.lambda_tv * loss_tv
            chainer.report({'loss_tv': loss_tv}, gen)
 
        # discriminator error
        if self.args.lambda_dis>0:
            y_fake = dis(F.concat([x_in, x_out]))
            #batchsize,_,w,h = y_fake.data.shape
            #loss_dis = F.sum(F.softplus(-y_fake)) / batchsize / w / h
            loss_dis = self.loss_func_comp(y_fake,1.0)
            chainer.report({'loss_dis': loss_dis}, gen)
            loss_gen += self.args.lambda_dis * loss_dis

        # update generator model
        gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        ## discriminator
        if self.args.lambda_dis>0:
            y_real = dis(F.concat([x_in, t_out]))
            loss_real = self.loss_func_comp(y_real,1.0)
            y_fake = dis(x_in_out_copy)
            loss_fake = self.loss_func_comp(y_fake,0.0)
            ## mis-matched input-output pair should be discriminated as fake
            if self._buffer.num_imgs > 40:
                f_in = self.gen.xp.concatenate(random.sample(self._buffer.images, len(x_in)))
                f_in = Variable(f_in[:,:x_in.shape[1],:,:])
                loss_mispair = self.loss_func_comp(dis(F.concat([f_in,t_out])),0.0)
            else:
                loss_mispair = 0
            loss = loss_fake + loss_real + loss_mispair
            dis.cleargrads()
            loss.backward()
            dis_optimizer.update()
            chainer.report({'loss_fake': loss_fake}, dis)
            chainer.report({'loss_real': loss_real}, dis)
            chainer.report({'loss_mispair': loss_mispair}, dis)
