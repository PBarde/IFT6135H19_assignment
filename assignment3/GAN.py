import torch
from torch import nn
from torch.autograd import  Variable, grad

# This code is inspired by:
# - https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py
# - https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class Generator(nn.Module):

    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(in_features=z_size, out_features=256)
        self.linear_activation = nn.ELU()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3), padding=(4, 4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), padding=(1, 1))
        )

        self.dgan_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        )

        self.output_activation = nn.Tanh()

    def forward(self, z):
        h = torch.unsqueeze(torch.unsqueeze(z, -1), -1)

        h = self.dgan_conv_stack(h)
        return self.output_activation(h)

    def loss(self, fake):
        return -torch.mean(fake)


class Discriminator(nn.Module):

    def __init__(self, im_size, device):
        super(Discriminator, self).__init__()

        flatten_size = im_size*im_size*3
        self.mlp_stack = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=flatten_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

        self.device = device

    def forward(self, x):
        h = self.conv_stack(x)
        h = h.view(-1)
        return h

    def clipWeights(self, c):
        for w in self.parameters():
            w.data.clamp_(-c, c)

    def gradientPenality(self, real, fake):
        shape = [real.size(0)] + [1] * (real.dim() - 1)
        e = Variable(torch.rand(shape, device=self.device))

        x_hat = e*real + (1-e)*fake
        x_hat = Variable(x_hat, requires_grad=True).to(self.device)
        d_hat = self(x_hat)
        d_grad = grad(d_hat, x_hat, torch.ones(d_hat.shape, device=self.device), create_graph=True)[0].view(x_hat.size(0), -1)

        return torch.mean((d_grad.norm(p=2, dim=1) - 1)**2)

    def loss(self, real, fake):
        d_real = self(real)
        d_fake = self(fake)
        w_distance = torch.mean(d_real).sub(torch.mean(d_fake))

        gp = self.gradientPenality(real, fake)
        return -w_distance + 10*gp

def DGAN_initialization(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, 0, 0.02)