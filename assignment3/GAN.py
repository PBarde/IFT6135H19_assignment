import torch
from torch import nn
from torch.autograd import  Variable, grad
from torch import optim
from classify_svhn import get_data_loader

# This code is inspired by: https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_pytorch.py

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

        self.output_activation = nn.Tanh()

    def forward(self, z):
        h = self.linear(z)
        h = self.linear_activation(h)
        h = torch.unsqueeze(torch.unsqueeze(h, -1), -1)

        h = self.conv_stack(h)
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

        self.device = device

    def forward(self, x):
        return self.mlp_stack(x)

    def clipWeights(self, c):
        for w in self.parameters():
            w.data.clamp_(-c, c)

    def gradientPenality(self, real, fake):
        shape = [real.size(0)] + [1] * (real.dim() - 1)
        e = Variable(torch.rand(shape, device=self.device))

        x_hat = e*real + (1-e)*fake
        x_hat = Variable(x_hat, requires_grad=True).to(device)
        d_hat = self(x_hat)
        d_grad = grad(d_hat, x_hat, torch.ones(d_hat.shape, device=self.device), create_graph=True)[0].view(x_hat.size(0), -1)

        return torch.mean((d_grad.norm(p=2, dim=1) - 1)**2)

    def loss(self, real, fake):
        d_real = self(real)
        d_fake = self(fake)
        w_distance = torch.mean(d_real).sub(torch.mean(d_fake))

        gp = self.gradientPenality(real, fake)
        return -w_distance + 10*gp


if __name__ == "__main__":
    lr = 0.00005
    batch_size = 16
    cliping = 0.01
    z_size = 100
    im_size = 32
    n_critic = 5

    device = torch.device("cpu")

    g = Generator(z_size).to(device)
    d = Discriminator(im_size, device).to(device)

    g_optim = optim.Adam(g.parameters(), lr=lr)
    d_optim = optim.Adam(d.parameters(), lr=lr)

    train, valid, test = get_data_loader("svhn", batch_size)

    for i in range(10):

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

            d_loss = d.loss(real_sample, fake_sample)
            d_loss.backward()
            d_optim.step()

        # Train the generator
        d_optim.zero_grad()
        g_optim.zero_grad()

        z = Variable(torch.randn(batch_size, z_size, device=device))

        fake_sample = g(z)

        g_loss = g.loss(d(fake_sample))
        g_loss.backward()
        g_optim.step()