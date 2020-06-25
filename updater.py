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
                if random.choice([True, False]):
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
        self.xp = self.gen.xp
        self._buffer = ImagePool(50 * self.args.batch_size)

    def loss_func_comp(self,y, val, noise=0, lambda_reg=1.0):
        xp = cuda.get_array_module(y.data)
        if noise>0:
            val += random.normalvariate(0,noise)   ## jitter for the target value
    #        val += random.uniform(-noise, noise)   ## jitter for the target value
        shape = y.data.shape
        if y.shape[1] == 2:
            shape = (shape[0],1,shape[2],shape[3])
            target = xp.full(shape, val, dtype=y.dtype)
            W = F.sigmoid(y[:,1,:,:])
            loss = F.average( ((y[:,0,:,:]-target)**2) * W )  ## weighted loss
            return loss + lambda_reg * F.mean_squared_error(W,xp.ones(W.shape,dtype=W.dtype))
        else:
            target = xp.full(shape, val, dtype=y.dtype)
            return F.mean_squared_error(y, target)

    def total_variation(self,x,tau=1e-6):
        xp = cuda.get_array_module(x.data)
        wh = xp.tile(xp.asarray([[[[1,0],[-1,0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
        ww = xp.tile(xp.asarray([[[[1, -1],[0, 0]]]], dtype=x.dtype),(x.data.shape[1],1,1))
        dx = F.convolution_2d(x, W=wh)
        dy = F.convolution_2d(x, W=ww)
        d = F.sqrt(dx**2 + dy**2 + xp.full(dx.data.shape, tau**2, dtype=dx.dtype))
        return(F.average(d))

    def total_variation2(self,x,tau=None):
        xp = cuda.get_array_module(x.data)
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return F.average(F.absolute(dx))+F.average(F.absolute(dy))

    ## multi-class focal loss
    def softmax_focalloss(self, x, t, class_num=4, gamma=2, eps=1e-7):
        p = F.clip(F.softmax(x), x_min=eps, x_max=1-eps)
#        print(p.shape,self.xp.eye(class_num)[t[:,0,:,:]].shape)
        q = -t * F.log(p)
        return F.sum(q * ((1 - p) ** gamma))

    def update_core(self):        
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis

        ## image conversion
        batch = self.get_iterator('main').next()
        x_in, t_out = self.converter(batch, self.device)
        x_in = Variable(x_in)
        x_out = gen(add_noise(x_in, sigma=self.args.noise))
        x_in_out = F.concat([x_in,x_out])
#        print(x_in.shape,x_out.shape, t_out.shape)

        loss_gen=0
        # reconstruction error
        if self.args.lambda_rec_l1>0:
            loss_rec_l1 = F.mean_absolute_error(x_out, t_out)
            loss_gen = loss_gen + self.args.lambda_rec_l1*loss_rec_l1       
            chainer.report({'loss_L1': loss_rec_l1}, gen)
        if self.args.lambda_rec_l2>0:
            loss_rec_l2 = F.mean_squared_error(x_out, t_out)
            loss_gen = loss_gen + self.args.lambda_rec_l2*loss_rec_l2
            chainer.report({'loss_L2': loss_rec_l2}, gen)
        if self.args.lambda_rec_ce>0:
            loss_rec_ce = self.softmax_focalloss(x_out, t_out)
            loss_gen = loss_gen + self.args.lambda_rec_ce*loss_rec_ce
            chainer.report({'loss_CE': loss_rec_ce}, gen)

        # total variation
        if self.args.lambda_tv > 0:
            loss_tv = self.total_variation2(x_out, self.args.tv_tau)
            loss_gen = loss_gen + self.args.lambda_tv * loss_tv
            chainer.report({'loss_tv': loss_tv}, gen)
 
        # Adversarial loss
        if self.args.lambda_dis>0:
            y_fake = dis(x_in_out)
            if self.args.dis_wgan:
                loss_adv = -F.average(y_fake)
            else:
                #batchsize,_,w,h = y_fake.data.shape
                #loss_dis = F.sum(F.softplus(-y_fake)) / batchsize / w / h
                loss_adv = self.loss_func_comp(y_fake,1.0,self.args.dis_jitter)
            chainer.report({'loss_dis': loss_adv}, gen)
            loss_gen = loss_gen + self.args.lambda_dis * loss_adv

        # update generator model
        gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update(loss=loss_gen)

        ## discriminator
        if self.args.lambda_dis>0:
            x_in_out_copy = self._buffer.query(x_in_out.array)
            if self.args.dis_wgan: ## synthesised -, real +
                eps = self.xp.random.uniform(0, 1, size=len(batch)).astype(self.xp.float32)[:, None, None, None]
                loss_real = -F.average(dis(F.concat([x_in, t_out])))
                loss_fake = F.average(dis(x_in_out_copy))
                y_mid = eps * x_in_out + (1.0 - eps) * x_in_out_copy
                # gradient penalty
                gd, = chainer.grad([dis(y_mid)], [y_mid], enable_double_backprop=True)
                gd = F.sqrt(F.batch_l2_norm_squared(gd) + 1e-6)
                loss_dis_gp = F.mean_squared_error(gd, self.xp.ones_like(gd.data))                
                chainer.report({'loss_gp': self.args.lambda_wgan_gp * loss_dis_gp}, dis)
                loss_dis = (loss_fake + loss_real) * 0.5 + self.args.lambda_wgan_gp * loss_dis_gp
            else:
                loss_real = self.loss_func_comp(dis(F.concat([x_in, t_out])),1.0,self.args.dis_jitter)
                loss_fake = self.loss_func_comp(dis(x_in_out_copy),0.0,self.args.dis_jitter)
                ## mis-matched input-output pair should be discriminated as fake
                if self._buffer.num_imgs > 40 and self.args.lambda_mispair>0:
                    f_in = self.gen.xp.concatenate(random.sample(self._buffer.images, len(x_in)))
                    f_in = Variable(f_in[:,:x_in.shape[1],:,:])
                    loss_mispair = self.loss_func_comp(dis(F.concat([f_in,t_out])),0.0,self.args.dis_jitter)
                    chainer.report({'loss_mispair': loss_mispair}, dis)
                else:
                    loss_mispair = 0
                loss_dis = 0.5*(loss_fake + loss_real) + self.args.lambda_mispair * loss_mispair

            # common for discriminator
            chainer.report({'loss_fake': loss_fake}, dis)
            chainer.report({'loss_real': loss_real}, dis)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update(loss=loss_dis)
