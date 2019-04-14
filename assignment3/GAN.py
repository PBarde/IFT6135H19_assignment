import torch
from torch import nn
from torch import functional as f
from torch.autograd import  Variable
from torch import optim
from classify_svhn import get_data_loader

import matplotlib.pyplot as plt


# This code is inspired by: https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py

lr = 0.00005
batch_size = 16
cliping = 0.01
z_size = 100
im_size = 32
n_critic = 5

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class Generator(nn.Module):

    def __init__(self):
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

    def forward(self, z):
        h = self.linear(z)
        h = self.linear_activation(h)
        h = torch.unsqueeze(torch.unsqueeze(h, -1), -1)

        h = self.conv_stack(h)
        return h

    @staticmethod
    def loss(fake):
        return -torch.mean(fake)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        flatten_size = im_size*im_size*3
        self.mlp_stack = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=flatten_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )


    def forward(self, x):
        return self.mlp_stack(x)

    def clipWeights(self):
        for w in self.parameters():
            w.data.clamp_(-cliping, cliping)

    @staticmethod
    def loss(real, fake):
        return -torch.mean(real).sub(torch.mean(fake))

def showImg(x):
    x = x.permute(1, 2, 0)
    plt.imshow((x.numpy() * 0.5) + 0.5)



if __name__ == "__main__":
    device = torch.device("cpu")

    cuda = torch.cuda.is_available()
    if False:
        device = torch.device("cuda")

    g = Generator().to(device)
    d = Discriminator().to(device)

    g_optim = optim.RMSprop(g.parameters(), lr=lr)
    d_optim = optim.RMSprop(d.parameters(), lr=lr)

    train, valid, test = get_data_loader("svhn", batch_size)

    for i in range(10000):

        # Train more the dicriminator
        for j in range(n_critic):
            d_optim.zero_grad()
            g_optim.zero_grad()

            z = Variable(torch.randn(batch_size, z_size, device=device))
            fake_sample = g(z)

            # get a batch of real sample
            dataloader_iterator = iter(train)
            try:
                real_sample, target = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train)
                real_sample, target = next(dataloader_iterator)

            real_sample = real_sample.to(device)

            d_loss = d.loss(d(real_sample), d(fake_sample))
            d_loss.backward()
            d_optim.step()

            d.clipWeights()

        # Train the generator
        d_optim.zero_grad()
        g_optim.zero_grad()

        z = Variable(torch.randn(batch_size, z_size, device=device))

        fake_sample = g(z)

        g_loss = g.loss(d(fake_sample))
        g_loss.backward()
        g_optim.step()