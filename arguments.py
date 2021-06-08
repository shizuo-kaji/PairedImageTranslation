import argparse
import numpy as np
import chainer.functions as F
from consts import activation_func,dtypes,norm_layer,optim
import os
import json,codecs

default_values = {'out': 'result', 'root': 'data', 'btoa': False, 'train': '__train__', 'val': '__test__', 'from_col': [0], 'to_col': [1], 'imgtype': 'jpg', \
    'crop_width': None, 'crop_height': None, 'grey': None, 'clipA': [None,None], 'clipB': [None,None], 'class_num': 0, 'class_weight': None, 'focal_gamma': 2, 'stack': 1, \
    'batch_size': 1, 'epoch': 400, 'gpu': 0, \
    'learning_rate': None, 'learning_rate_gen': 2e-4, 'learning_rate_dis': 1e-4, 'lr_drop': 1, \
    'weight_decay': 1e-7, 'weight_decay_norm': 'l2', \
    'snapinterval': -1, 'display_interval': 100, 'nvis': 3, 'vis_freq': None, 'parameter_statistics': False, \
    'lambda_rec_l1': 10, 'lambda_rec_l2': 0, 'lambda_rec_ce': 0, 'lambda_dis': 1, 'lambda_tv': 0, 'lambda_reg': 0, \
    'lambda_mispair': 0, 'lambda_wgan_gp': 10, 'tv_tau': 1e-3, 'loss_ksize': 1, \
    'random_translate': 4, 'random_rotation': 0, 'random_scale': 0, 'noise': 0, 'noise_z': 0, \
    'load_optimizer': False, 'optimizer': 'Adam', \
    'dtype': 'fp32', 'eqconv': False, 'spconv': False, 'senet': False, \
    'dis_activation': 'lrelu', 'dis_out_activation': 'none', 'dis_ksize': 4, 'dis_chs': None, \
    'dis_basech': 64, 'dis_ndown': 3, 'dis_down': 'down', 'dis_sample': 'down', 'dis_jitter': 0.2, 'dis_dropout': None, \
    'dis_norm': 'batch_aff', 'dis_reg_weighting': 0, 'dis_attention': False, 'dis_warmup': -1, \
    'gen_pretrained_encoder': '', 'gen_pretrained_lr_ratio': 0, 'gen_activation': 'relu', 'gen_out_activation': 'tanh', 'gen_chs': None, \
    'gen_ndown': 3, 'gen_basech': 64, 'gen_fc': 0, 'gen_fc_activation': 'relu', 'gen_nblock': 9, 'gen_ksize': 3, \
    'gen_sample': 'none-7', 'gen_down': 'down', 'gen_up': 'unpool', 'gen_dropout': None, 'gen_norm': 'batch_aff', \
    'unet': 'conv', 'skipdim': 4, 'latent_dim': -1, \
    'ch': None, 'out_ch': None}


def arguments():
    parser = argparse.ArgumentParser(description='Image-to-image translation using a paired training dataset')
    parser.add_argument('--argfile', '-a', help="specify args file to read")
    parser.add_argument('--out', '-o', help='Directory to output the result')
    # input image
    parser.add_argument('--root', '-R', help='directory containing image files')
    parser.add_argument('--btoa', action='store_true', help='convert in the opposite way (B to A)')
    parser.add_argument('--train', '-t', help='text file containing image pair filenames for training')
    parser.add_argument('--val', help='text file containing image pair filenames for validation')
    parser.add_argument('--from_col', '-c1', type=int, nargs="*", help='column index of FromImage')
    parser.add_argument('--to_col', '-c2', type=int, nargs="*", help='column index of ToImage')
    parser.add_argument('--imgtype', '-it', help="image file type (file extension)")
    parser.add_argument('--crop_width', '-cw', type=int, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--crop_height', '-ch', type=int, help='this value may have to be divisible by a large power of two (if you encounter errors)')
    parser.add_argument('--grey', action='store_true', help='load image (jpg/png) in greyscale')
    parser.add_argument('--clipA', '-ca', type=float, nargs=2, help="lower and upper limit for pixel values of images in domain A")
    parser.add_argument('--clipB', '-cb', type=float, nargs=2, help="lower and upper limit for pixel values of images in domain B")
    parser.add_argument('--class_num', '-cn', type=int, help='number of classes for pixelwise classification (only for images in domain B)')
    parser.add_argument('--class_weight', type=float, nargs="*", help='weight for each class for pixelwise classification (only for images in domain B)')
    parser.add_argument('--stack', type=int, help='number of images in a stack (>1 means 2.5D)')

    # training
    parser.add_argument('--batch_size', '-b', type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--learning_rate', '-lr', type=float, help='Learning rate')
    parser.add_argument('--learning_rate_gen', '-lrg', type=float)
    parser.add_argument('--learning_rate_dis', '-lrd', type=float)
    parser.add_argument('--lr_drop', type=int, help='How many times the learning rate drops')
    parser.add_argument('--weight_decay', '-wd', type=float, help='weight decay for regularization')
    parser.add_argument('--weight_decay_norm', '-wn', choices=['l1','l2'], help='norm of weight decay for regularization')

    # snapshot and evaluation
    parser.add_argument('--snapinterval', '-si', type=int, help='take snapshot every this epoch')
    parser.add_argument('--display_interval', type=int, help='Interval of displaying log to console')
    parser.add_argument('--nvis', type=int, help='number of images in visualisation after each epoch')
    parser.add_argument('--vis_freq', '-vf', type=int, help='visualisation frequency in iteration')
    parser.add_argument('--parameter_statistics', '-ps', action='store_true',
                        help='Log NN parameter statistics (very slow)')

    # weights
    parser.add_argument('--lambda_rec_l1', '-l1', type=float, help='weight for L1 reconstruction loss')
    parser.add_argument('--lambda_rec_l2', '-l2', type=float, help='weight for L2 reconstruction loss')
    parser.add_argument('--lambda_rec_ce', '-lce', type=float, help='weight for softmax focal reconstruction loss')
    parser.add_argument('--lambda_dis', '-ldis', type=float, help='weight for adversarial loss')
    parser.add_argument('--lambda_tv', '-ltv', type=float, help='weight for total variation')
    parser.add_argument('--lambda_reg', '-lreg', type=float, help='weight for regularisation for encoders')
    parser.add_argument('--lambda_mispair', '-lm', type=float, help='weight for discriminator rejecting mis-matched (real,real) pairs')
    parser.add_argument('--lambda_wgan_gp', '-lwgp', type=float, help='lambda for the gradient penalty for WGAN')
    parser.add_argument('--tv_tau', '-tt', type=float, help='smoothing parameter for total variation')
    parser.add_argument('--focal_gamma', '-fg', type=float, help='gamma for the focal loss')
    parser.add_argument('--loss_ksize', '-lk', type=int, help='take average pooling of this kernel size before computing L1 and L2 losses')

    # data augmentation
    parser.add_argument('--random_translate', '-rt', type=int, help='jitter input images by random translation')
    parser.add_argument('--random_rotation', '-rr', type=int, help='jitter input images by random rotation (in degree)')
    parser.add_argument('--random_scale', '-rs', type=float, help='jitter input images by random scaling (in ratio)')
    parser.add_argument('--noise', '-n', type=float, help='strength of noise injection')
    parser.add_argument('--noise_z', '-nz', type=float, help='strength of noise injection for the latent variable')

    # load model/optimizer
    parser.add_argument('--load_optimizer', '-mo', action='store_true', help='load optimizer parameters')
    parser.add_argument('--model_gen', '-m', default='', help='specify a learnt encoder/generator model file')
    parser.add_argument('--model_dis', '-md', default='', help='specify a learnt discriminator model file')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(), help='optimizer')

    # network
    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), help='floating point precision')
    parser.add_argument('--eqconv', '-eq', action='store_true', help='Equalised Convolution')
    parser.add_argument('--spconv', '-sp', action='store_true', help='Separable Convolution')
    parser.add_argument('--senet', '-se', action='store_true', help='Enable Squeeze-and-Excitation mechanism')

    # discriminator
    parser.add_argument('--dis_activation', '-da', choices=activation_func.keys(), help='activation of middle layers discriminators')
    parser.add_argument('--dis_out_activation', '-do', choices=activation_func.keys(), help='activation of last layer of discriminators')
    parser.add_argument('--dis_ksize', '-dk', type=int, help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_chs', '-dc', type=int, nargs="*", help='Number of channels in down layers in discriminator')
    parser.add_argument('--dis_basech', '-db', type=int, help='the base number of channels in discriminator (doubled in each down-layer)')
    parser.add_argument('--dis_ndown', '-dl', type=int, help='number of down layers in discriminator')
    parser.add_argument('--dis_down', '-dd', help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', choices=norm_layer, help='nomalisation layer for discriminator')
    parser.add_argument('--dis_reg_weighting', '-dw', type=float, help='regularisation of weighted discriminator. Set 0 to disable weighting')
    parser.add_argument('--dis_wgan', '-wgan', action='store_true',help='WGAN-GP')
    parser.add_argument('--dis_attention', action='store_true',help='attention mechanism for discriminator')
    parser.add_argument('--dis_warmup', type=int, help='number of warm-up iterations before discriminator starts to learn')

    # generator    
    parser.add_argument('--gen_pretrained_encoder', '-gp', type=str, choices=["","vgg","resnet"], help='Use pretrained ResNet/VGG as encoder')
    parser.add_argument('--gen_pretrained_lr_ratio', '-gpr', type=float, help='learning rate multiplier for the pretrained part')
    parser.add_argument('--gen_activation', '-ga', choices=activation_func.keys(), help='activation for middle layers of generators')
    parser.add_argument('--gen_out_activation', '-go', choices=activation_func.keys(), help='activation for last layers of generators')
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", help='Number of channels in down layers in generator')
    parser.add_argument('--gen_ndown', '-gl', type=int, help='number of down layers in generator')
    parser.add_argument('--gen_basech', '-gb', type=int, help='the base number of channels in generator (doubled in each down-layer)')
    parser.add_argument('--gen_fc', '-gfc', type=int, help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_fc_activation', '-gfca', choices=activation_func.keys(), help='activation of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-gnb', type=int, help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', choices=norm_layer, help='nomalisation layer for generator')
    parser.add_argument('--unet', '-u', help='use u-net for generator')
    parser.add_argument('--skipdim', '-sd', type=int, help='channel number for skip connections')
    parser.add_argument('--latent_dim', type=int, help='dimension of the latent space between encoder and decoder')

    ####
    args = parser.parse_args()
    # number of channels in input/output images: infered from data or args file.
    args.ch = None
    args.out_ch = None

    ## set default values from file 
    if args.argfile:
        with open(args.argfile, 'r') as f:
            larg = json.load(f)
    else:
        larg = []
    for x in vars(args):
        if getattr(args, x) is None:
            if x in larg:
                setattr(args, x, larg[x])
            elif x in default_values:
                setattr(args, x, default_values[x])

    if args.learning_rate:
        args.learning_rate_gen = args.learning_rate
        args.learning_rate_dis = args.learning_rate/2
    if "resnet" in args.gen_pretrained_encoder:
        args.gen_chs = [64,256,512,1024,2048][:args.gen_ndown]
    elif "vgg" in args.gen_pretrained_encoder:
        args.gen_chs = [64,128,256,512,512][:args.gen_ndown]
    if not args.gen_chs:
        args.gen_chs = [int(args.gen_basech) * (2**i) for i in range(args.gen_ndown)]
    else:
        args.gen_ndown = len(args.gen_chs)
    if not args.dis_chs:
        args.dis_chs = [int(args.dis_basech) * (2**i) for i in range(args.dis_ndown)]
    else:
        args.dis_ndown = len(args.dis_chs)

    if args.imgtype=="dcm":
        args.grey = True
        if args.clipA[0] is None:
            args.clipA = [-1024,2000]
        if args.clipB[0] is None:
            args.clipB = [-1024,2000]
    elif args.imgtype not in ['csv','txt','npy']:
        if args.clipA[0] is None:
            args.clipA = [0,255]
        if args.clipB[0] is None:
            args.clipB = [0,255]

    if args.gen_fc>0 and args.crop_width is None:
        print("Specify crop_width and crop_height!")
        exit()

    if args.class_num>0:
        args.gen_out_activation='none'
        print("the last activation is set to stack-wise softmax for point-wise classification.")
        if args.out_ch is None:
            args.out_ch = args.class_num

    # convert.py
    if args.out_ch is None:
        args.out_ch = 1 if args.grey else 3
    if args.ch is None:
        args.ch = 1 if args.grey else 3

    print(args)

    return(args)

