from __future__ import print_function
import argparse,os
import numpy as np

import random


parser = argparse.ArgumentParser(description='')
parser.add_argument('input', help='input')
parser.add_argument('--ratio', '-r', default=0.8, type=float, help='train ratio')
args = parser.parse_args()

fn,ext = os.path.splitext(args.input)

with open(args.input, "r") as f:
    data = f.read().split('\n')

if data[-1]=="":
    data.pop(-1)
    
random.shuffle(data)
n = int(args.ratio*len(data))

with open(fn+"_train.txt", "w") as f:
    f.write("\n".join(data[:n]))

with open(fn+"_val.txt", "w") as f:
    f.write("\n".join(data[n:]))

