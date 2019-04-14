from __future__ import print_function
import torch
import torch.utils.data
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.encoder_activation1 = nn.ELU()
        self.encoder_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.encoder_activation2 = nn.ELU()
        self.encoder_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5))
        self.encoder_activation3 = nn.ELU()
        self.encoder_linear_mean = nn.Linear(in_features=256, out_features=100)
        self.encoder_linear_var = nn.Linear(in_features=256, out_features=100)

        # Decoder
        self.decoder_linear = nn.Linear(in_features=100, out_features=256)
        self.decoder_activation1 = nn.ELU()
        self.decoder_conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), padding=(4, 4))
        self.decoder_activation2 = nn.ELU()
        self.decoder_upsampling1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(2, 2))
        self.decoder_activation3 = nn.ELU()
        self.decoder_upsampling2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(2, 2))
        self.decoder_activation4 = nn.ELU()
        self.decoder_conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=(2, 2))

    def encode(self, x):
        h = nn.Sequential(
            self.encoder_conv1,
            self.encoder_activation1,
            self.encoder_pool1,
            self.encoder_conv2,
            self.encoder_activation2,
            self.encoder_pool2,
            self.encoder_conv3,
            self.encoder_activation3
        )(x)
        h = torch.squeeze(h)
        z_mu = self.encoder_linear_mean(h)
        z_sig2 = self.encoder_linear_var(h)
        return z_mu, z_sig2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_linear(z)
        h = self.decoder_activation1(h)
        h = torch.unsqueeze(torch.unsqueeze(h, -1), -1)
        h = nn.Sequential(
            self.decoder_conv1,
            self.decoder_activation2,
            self.decoder_upsampling1,
            self.decoder_conv2,
            self.decoder_activation3,
            self.decoder_upsampling2,
            self.decoder_conv3,
            self.decoder_activation4,
            self.decoder_conv4
        )(h)
        o = torch.sigmoid(h)
        return o

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
