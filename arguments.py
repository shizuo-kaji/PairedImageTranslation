import argparse
import numpy as np
import chainer.functions as F
from consts import activation,dtypes
import os
from datetime import datetime as dt
from chainerui.utils import save_args

def arguments():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--train', '-t', help='text file containing image pair filenames for training')
    parser.add_argument('--val', help='text file containing image pair filenames for validation')
    parser.add_argument('--imgtype', '-it', default="jpg", help="image file type (file extension)")
    parser.add_argument('--argfile', '-a', help="specify args file to read")
    parser.add_argument('--from_col', '-c1', type=int, nargs="*", default=[0],
                        help='column index of FromImage')
    parser.add_argument('--to_col', '-c2', type=int, nargs="*", default=[1],
                        help='column index of ToImage')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=400,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default='.',
                        help='directory containing image files')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)

    parser.add_argument('--snapinterval', '-si', type=int, default=-1, 
                        help='take snapshot every this epoch')
    parser.add_argument('--display_interval', type=int, default=500,
                        help='Interval of displaying log to console')
    parser.add_argument('--nvis', type=int, default=3,
                        help='number of images in visualisation after each epoch')

    parser.add_argument('--crop_width', '-cw', type=int, default=128, help='better to have a value divisible by a large power of two')
    parser.add_argument('--crop_height', '-ch', type=int, default=128, help='better to have a value divisible by a large power of two')
    parser.add_argument('--grey', action='store_true',
                        help='greyscale')

    parser.add_argument('--lambda_rec_l1', '-l1', type=float, default=1.0)
    parser.add_argument('--lambda_rec_l2', '-l2', type=float, default=0.0)
    parser.add_argument('--lambda_dis', '-ldis', type=float, default=0.1)
    parser.add_argument('--lambda_tv', '-ltv', type=float, default=0.0)
    parser.add_argument('--lambda_mispair', '-lm', type=float, default=1.0)
    parser.add_argument('--tv_tau', '-tt', type=float, default=1e-3,
                        help='smoothing parameter for total variation')

    parser.add_argument('--load_optimizer', '-op', action='store_true', help='load optimizer parameters')
    parser.add_argument('--model_gen', '-m', default='')
    parser.add_argument('--model_dis', '-md', default='')

    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    parser.add_argument('--eqconv', '-eq', action='store_true',
                        help='Equalised Convolution')
    parser.add_argument('--spconv', '-sp', action='store_true',
                        help='Separable Convolution')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0,  #default:  1e-7
                        help='weight decay for regularization')
    parser.add_argument('--weight_decay_norm', '-wn', choices=['l1','l2'], default='l2',
                        help='norm of weight decay for regularization')
    parser.add_argument('--vis_freq', '-vf', type=int, default=4000,
                        help='visualisation frequency in iteration')

    # data augmentation
    parser.add_argument('--random', '-rt', default=True, help='random flip/crop')
    parser.add_argument('--noise', '-n', type=float, default=0, help='strength of noise injection')
    parser.add_argument('--noise_z', '-nz', type=float, default=0,
                        help='strength of noise injection for the latent variable')

    # discriminator
    parser.add_argument('--dis_activation', '-da', default='lrelu', choices=activation.keys())
    parser.add_argument('--dis_basech', '-db', type=int, default=64,
                        help='the base number of channels in discriminator')
    parser.add_argument('--dis_ksize', '-dk', type=int, default=4,    # default 4
                        help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_ndown', '-dl', type=int, default=3,    # default 3
                        help='number of down layers in discriminator')
    parser.add_argument('--dis_down', '-dd', default='down', choices=['down','maxpool','maxpool_res','avgpool','avgpool_res','none'],  ## default down
                        help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', default='none',          ## default down
                        help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, default=0,
                        help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, default=None, 
                        help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', default='instance',
                        choices=['instance', 'batch','batch_aff', 'rbatch', 'fnorm', 'none'])

    # generator: G: A -> B, F: B -> A
    parser.add_argument('--gen_activation', '-ga', default='relu', choices=activation.keys())
    parser.add_argument('--gen_fc_activation', '-gfca', default='relu', choices=activation.keys())
    parser.add_argument('--gen_out_activation', '-go', default='tanh', choices=activation.keys())
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", default=[64,128,256,512],
                        help='Number of channels in down layers in generator; the first entry should coincide with the number of channels in the input images')
    parser.add_argument('--gen_fc', '-gfc', type=int, default=0,
                        help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-nb', type=int, default=9,  # default 9
                        help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, default=3,    # default 4
                        help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', default='none',
                        help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', default='down', choices=['down','maxpool','maxpool_res','avgpool','avgpool_res','none'],
                        help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', default='resize', choices=['unpool','unpool_res','deconv','pixsh','resize','resize_res','none'],
                        help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, default=None, 
                        help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', default='instance',
                        choices=['instance', 'batch','batch_aff', 'rbatch', 'fnorm', 'none'])
    parser.add_argument('--unet', '-u', default='none', choices=['none','no_last','with_last'],
                        help='use u-net for generator')

    args = parser.parse_args()
    args.out = os.path.join(args.out, dt.now().strftime('%m%d_%H%M')+"_cgan")
    save_args(args, args.out)
    print(args)
    print(args.out)

    args.wgan=False
    args.dtype = dtypes[args.dtype]
    args.dis_activation = activation[args.dis_activation]
    args.gen_activation = activation[args.gen_activation]
    args.gen_fc_activation = activation[args.gen_fc_activation]
    args.gen_out_activation = activation[args.gen_out_activation]
    args.lrdecay_start = args.epoch//2
    args.lrdecay_period = args.epoch - args.lrdecay_start
    return(args)

