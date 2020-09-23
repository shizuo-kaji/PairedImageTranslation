import argparse
import numpy as np
import chainer.functions as F
from consts import activation_func,dtypes,norm_layer,optim
import os
from datetime import datetime as dt
from chainerui.utils import save_args

def arguments():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--btoa', action='store_true', help='convert in the opposite way (B to A)')
    parser.add_argument('--argfile', '-a', help="specify args file to read")
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    # input image
    parser.add_argument('--root', '-R', default='.',
                        help='directory containing image files')
    parser.add_argument('--train', '-t', default="__train__", help='text file containing image pair filenames for training')
    parser.add_argument('--val', default="__test__", help='text file containing image pair filenames for validation')
    parser.add_argument('--from_col', '-c1', type=int, nargs="*", default=[0],
                        help='column index of FromImage')
    parser.add_argument('--to_col', '-c2', type=int, nargs="*", default=[1],
                        help='column index of ToImage')
    parser.add_argument('--imgtype', '-it', default="jpg", help="image file type (file extension)")
    parser.add_argument('--crop_width', '-cw', type=int, default=None, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--crop_height', '-ch', type=int, default=None, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--grey', action='store_true', help='load image (jpg/png) in greyscale')
    parser.add_argument('--clip_below', '-cb', default=None, help="clip pixel value from below")
    parser.add_argument('--clip_above', '-ca', default=None, help="clip pixel value from above")    
    parser.add_argument('--class_num', '-cn', type=int,default=0, help='number of classes for pixelwise classification')

    # training
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=400,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--learning_rate_gen', '-lrg', type=float, default=2e-4)
    parser.add_argument('--learning_rate_dis', '-lrd', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-8,  #default:  1e-7
                        help='weight decay for regularization')
    parser.add_argument('--weight_decay_norm', '-wn', choices=['l1','l2'], default='l2',
                        help='norm of weight decay for regularization')

    # snapshot and evaluation
    parser.add_argument('--snapinterval', '-si', type=int, default=-1, 
                        help='take snapshot every this epoch')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--nvis', type=int, default=3,
                        help='number of images in visualisation after each epoch')
    parser.add_argument('--vis_freq', '-vf', type=int, default=None,
                        help='visualisation frequency in iteration')

    # weights
    parser.add_argument('--lambda_rec_l1', '-l1', type=float, default=10.0, help='weight for L1 reconstruction loss')
    parser.add_argument('--lambda_rec_l2', '-l2', type=float, default=0.0, help='weight for L2 reconstruction loss')
    parser.add_argument('--lambda_rec_ce', '-lce', type=float, default=0.0, help='weight for softmax focal reconstruction loss')
    parser.add_argument('--lambda_dis', '-ldis', type=float, default=1.0, help='weight for adversarial loss')
    parser.add_argument('--lambda_tv', '-ltv', type=float, default=0.0, help='weight for total variation')
    parser.add_argument('--lambda_mispair', '-lm', type=float, default=0, help='weight for discriminator rejecting mis-matched (real,real) pairs')
    parser.add_argument('--lambda_wgan_gp', '-lwgp', type=float, default=10,
                        help='lambda for the gradient penalty for WGAN')
    parser.add_argument('--tv_tau', '-tt', type=float, default=1e-3,
                        help='smoothing parameter for total variation')
    parser.add_argument('--loss_ksize', '-lk', type=int, default=1,
                        help='take average pooling of this kernel size before computing L1 and L2 losses')

    # data augmentation
    parser.add_argument('--random_translate', '-rt', type=int, default=4, help='jitter input images by random translation')
    parser.add_argument('--noise', '-n', type=float, default=0, help='strength of noise injection')
    parser.add_argument('--noise_z', '-nz', type=float, default=0,
                        help='strength of noise injection for the latent variable')

    # load model/optimizer
    parser.add_argument('--load_optimizer', '-mo', action='store_true', help='load optimizer parameters')
    parser.add_argument('--model_gen', '-m', default='')
    parser.add_argument('--model_dis', '-md', default='')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')

    # network
    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    parser.add_argument('--eqconv', '-eq', action='store_true',
                        help='Equalised Convolution')
    parser.add_argument('--spconv', '-sp', action='store_true',
                        help='Separable Convolution')
    parser.add_argument('--senet', '-se', action='store_true',
                        help='Enable Squeeze-and-Excitation mechanism')

    # discriminator
    parser.add_argument('--dis_activation', '-da', default='lrelu', choices=activation_func.keys())
    parser.add_argument('--dis_out_activation', '-do', default='none', choices=activation_func.keys())
    parser.add_argument('--dis_ksize', '-dk', type=int, default=4,    # default 4
                        help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_chs', '-dc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in discriminator')
    parser.add_argument('--dis_basech', '-db', type=int, default=64,
                        help='the base number of channels in discriminator (doubled in each down-layer)')
    parser.add_argument('--dis_ndown', '-dl', type=int, default=3,
                        help='number of down layers in discriminator')
    parser.add_argument('--dis_down', '-dd', default='down',
                        help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', default='down',
                        help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, default=0,
                        help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, default=None, 
                        help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--dis_reg_weighting', '-dw', type=float, default=0,
                        help='regularisation of weighted discriminator. Set 0 to disable weighting')
    parser.add_argument('--dis_wgan', '-wgan', action='store_true',help='WGAN-GP')
    parser.add_argument('--dis_attention', action='store_true',help='attention mechanism for discriminator')

    # generator
    parser.add_argument('--gen_activation', '-ga', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_fc_activation', '-gfca', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_out_activation', '-go', default='tanh', choices=activation_func.keys())
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in generator')
    parser.add_argument('--gen_ndown', '-gl', type=int, default=3,
                        help='number of down layers in generator')
    parser.add_argument('--gen_basech', '-gb', type=int, default=64,
                        help='the base number of channels in generator (doubled in each down-layer)')
    parser.add_argument('--gen_fc', '-gfc', type=int, default=0,
                        help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-gnb', type=int, default=9,
                        help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, default=3,
                        help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', default='none',
                        help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', default='down',
                        help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', default='resize',
                        help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, default=None, 
                        help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--unet', '-u', default='conv',
                        help='use u-net for generator')
    parser.add_argument('--skipdim', '-sd', type=int, default=4,
                        help='channel number for skip connections')

    ####
    args = parser.parse_args()
    args.out = os.path.join(args.out, dt.now().strftime('%m%d_%H%M')+"_cgan")

    if not args.gen_chs:
        args.gen_chs = [int(args.gen_basech) * (2**i) for i in range(args.gen_ndown)]
    if not args.dis_chs:
        args.dis_chs = [int(args.dis_basech) * (2**i) for i in range(args.dis_ndown)]
    if args.gen_fc>0 and args.crop_width is None:
        print("Specify crop_width and crop_height!")
        exit()
    save_args(args, args.out)
    print(args)
    print("\nresults are saved under: ",args.out)

    return(args)

