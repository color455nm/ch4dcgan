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

from dcgan import Generator
from dcgan import Discriminator
from dcgan import weights_init


parser = argparse.ArgumentParser(prog = "training routine DCGANs",
                                 description="Train a DCGAN network")
parser.add_argument("-c", "--config", required=True)
args = parser.parse_args()
configs = utils.load_json(args.config)
#with open(args.config, "r") as fn:
#    configs = json.load(fn)

CUDA = configs["CUDA"]
DATA_PATH = configs["DATA_PATH"]
OUT_PATH = configs["OUT_PATH"]
LOG_FILE = os.path.join(OUT_PATH, configs["LOG_FILE"])

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
utils.clear_folder(OUT_PATH)
utils.copy_file(args.config, OUT_PATH)
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

criterion = nn.BCELoss()

netG = Generator().to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# dataset MNIST 
dataset = dset.MNIST(root=DATA_PATH, download=True,
                     transform=transforms.Compose([
                         transforms.Resize(X_DIM),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))
                         ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=0)
"""
one can add pin_memory=True which will make sure data is stored at fixed GPU memory address and thus
increase the data loading speed during training.
"""

"""
The training procedure is basically
1. Train the discriminator with the real data and recognize it as real
2. Train the discriminator with fake data and recognize it as fake
3. Train the generator with the fake data and recognize it as real
"""

viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader):
        print("\rEpoch {} [{:05}/{}]".format(epoch,i,len(dataloader)), end=" ", flush=True)
        x_real = data[0].to(device)
        """we create real_label and fake_label tesnsor in real time because there is no guarantee
        that all sample batches will have the same size, the last batch is often smaller
        """
        real_label = torch.full((x_real.size(0),), REAL_LABEL, dtype=torch.float, device=device) 
        fake_label = torch.full((x_real.size(0),), FAKE_LABEL, dtype=torch.float, device=device)
        # update D with real data
        netD.zero_grad()
        y_real = netD(x_real)
        loss_D_real = criterion(y_real, real_label)
        loss_D_real.backward()

        # update D with fake data
        z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
        x_fake = netG(z_noise)
        y_fake = netD(x_fake.detach())
        loss_D_fake = criterion(y_fake, fake_label)
        loss_D_fake.backward()
        optimizerD.step()

        # update G with fake data
        netG.zero_grad()
        y_fake_r = netD(x_fake)
        loss_G = criterion(y_fake_r, real_label)
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('\nEpoch {} [{:05}/{}] loss_D_real:{:4f} loss_D_fake:{:.4f} loss_G:{:.4f}'.format(
                epoch, i, len(dataloader), loss_D_real.mean().item(), loss_D_fake.mean().item(),
                loss_G.mean().item()))
            vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
            with torch.no_grad():
                viz_sample = netG(viz_noise)
                vutils.save_image(viz_sample, os.path.join(OUT_PATH,'fake_sample_{}.png'.format(epoch)), normalize=True)
                torch.save(netG.state_dict(), os.path.join(OUT_PATH,'netG_{}.pth'.format(epoch)))
                torch.save(netD.state_dict(), os.path.join(OUT_PATH,'netD_{}.pth'.format(epoch)))


        break
