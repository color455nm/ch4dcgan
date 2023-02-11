import numpy as np
import torch
import torch.nn as nn

import utils


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, g_hidden: int = 64, image_channel: int = 1):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                # 1st layer
                nn.ConvTranspose2d(z_dim, g_hidden * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(g_hidden * 8),
                nn.ReLU(True),
                # 2nd layer
                nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4,  4, 2, 1, bias=False),
                nn.BatchNorm2d(g_hidden * 4),
                nn.ReLU(True),
                # 3rd layer
                nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2,  4, 2, 1, bias=False),
                nn.BatchNorm2d(g_hidden * 2),
                nn.ReLU(True),
                # 4th layer
                nn.ConvTranspose2d(g_hidden * 2, g_hidden,  4, 2, 1, bias=False),
                nn.BatchNorm2d(g_hidden),
                nn.ReLU(True),
                # output layer
                nn.ConvTranspose2d(g_hidden, image_channel,  4, 2, 1, bias=False),
                nn.Tanh()
                )

    def forward(self, x):
        return self.main(x)

def weights_init(m):
    """Helper function to initialize the network parameters. 
    Conv layers are initialized with Gausssian distribution.
    The affine oarameters (scaling factor) in batch normalization are also initialized
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
                

class Discriminator(nn.Module):
    def __init__(self, image_channel : int = 1, d_hidden: int = 64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(image_channel, d_hidden, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(d_hidden, d_hidden * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d_hidden * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 3rd layer
                nn.Conv2d(d_hidden * 2, d_hidden * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d_hidden * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4th layer
                nn.Conv2d(d_hidden * 4, d_hidden * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d_hidden * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.Conv2d(d_hidden * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
                

if __name__=="__main__":
    netG = Generator()
    netG.apply(weights_init)
    x = torch.randn(1, 100, 1, 1)
    z = netG(x)
    print("y size ", z.shape)

    netD = Discriminator().to('cpu')
    netD.apply(weights_init)
    # x = torch.randn(1, 64, 1, 1)
    y = netD(z)
    print(netG)
    print(netD)
