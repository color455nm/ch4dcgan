import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms 
import torchvision.utils as vutils

import utils

from datetime import datetime

from dcgan import Generator
from dcgan import Discriminator
from dcgan import weights_init


parser = argparse.ArgumentParser(prog = "Interpolation of DCGAN Generator",
                                 description="Interpolation DCGAN network")
parser.add_argument("-c", "--config", required=True)
args = parser.parse_args()
configs = utils.load_json(args.config)
#with open(args.config, "r") as fn:
#    configs = json.load(fn)

CUDA = configs["CUDA"]
DATA_PATH = configs["DATA_PATH"]
OUT_PATH = configs["OUT_PATH"]
LOG_FILE = os.path.join(
        OUT_PATH, 
        configs["LOG_FILE"].replace('.log',datetime.now().strftime("%Y%m%d-%H%M%S")+'.log'))

# Model parameters
BATCH_SIZE = configs["BATCH_SIZE"]
IMAGE_CHANNEL = configs["IMAGE_CHANNEL"]
Z_DIM = configs["Z_DIM"]
G_HIDDEN = configs["G_HIDDEN"]
X_DIM = configs["X_DIM"]
D_HIDDEN = configs["D_HIDDEN"]

# Training parameters
EPOCH_NUM = configs["EPOCH_NUM"]
REAL_LABEL = configs["REAL_LABEL"]
FAKE_LABEL = configs["FAKE_LABEL"]
lr = configs["lr"]
seed = configs["seed"]

# Set up
print("Logging to {}\n".format(LOG_FILE))
sys.stdout = utils.StdOut(LOG_FILE) # will redirect all messages from print to the log file 
# and show these messages in the console at the same time
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version {}".format(torch.__version__))

if CUDA:
    print("CUDA version {}".format(torch.version.cuda))
if seed is None:
    seed = np.random.randint(1, 10000)
print("Random Seed: ", seed)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
cudnn.benchmark = False # True
"""
cudnn.benchmark = True will tell cuDNN to choose the best set of algorithms for your model if the size of input
data is fixed; otherwise, cuDNN will have to find the best algorithm at each iterataion.
This will dramatically increase the GPU memory consumption, especially when your model architectures are changing during
training and you are doint both training and evaluation in your code.

"""
device = torch.device("cuda:0" if CUDA else "cpu") 

netG = Generator()
netG.load_state_dict(torch.load(os.path.join(OUT_PATH, 'netG_0.pth'))
netG.to(device)

if VIZ_MODE == 0:


