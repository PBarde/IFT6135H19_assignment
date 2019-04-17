from __future__ import print_function
import torch
import torch.utils.data
from torch import nn

# This code is inspired by: https://github.com/pytorch/examples/blob/master/vae/main.py
# and: https://github.com/bjlkeng/sandbox/tree/master/notebooks/variational_autoencoder-svhn


class VAE(nn.Module):
    def __init__(self, latent_size=100, type='MNIST'):
        super(VAE, self).__init__()
        self.type = type

        if type == 'MNIST':
            # Encoder
            self.encoder_sequence = nn.Sequential(
                # layer 1
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                # layer 2
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                # layer 3
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5)),
                nn.ELU()
            )
            self.encoder_linear_mean = nn.Linear(in_features=256, out_features=latent_size)
            self.encoder_linear_var = nn.Linear(in_features=256, out_features=latent_size)

            # Decoder
            # layer 1
            self.decoder_linear = nn.Linear(in_features=latent_size, out_features=256)
            self.decoder_activation1 = nn.ELU()
            self.decoder_sequence = nn.Sequential(
                # layer 2
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), padding=(4, 4)),
                nn.ELU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                # layer 3
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=(2, 2)),
                nn.ELU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                # layer 4
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=(2, 2)),
                nn.ELU(),
                # layer 5
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=(2, 2))
            )
            self.decoder_output_activation = nn.Sigmoid()

        else:
            input_size = 3072
            hidden_size = 2048
            dropout_p = 0.2
            # Encoder
            self.encoder_sequence = nn.Sequential(
                # input
                nn.BatchNorm1d(input_size),
                # layer 1
                nn.Linear(in_features=input_size, out_features=hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                # layer 2
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                # layer 3
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_p)
            )
            self.encoder_linear_mean = nn.Linear(in_features=hidden_size, out_features=latent_size)
            self.encoder_linear_var = nn.Linear(in_features=hidden_size, out_features=latent_size)

            # Decoder (same as GAN generator)
            self.decoder_sequence = nn.Sequential(
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
            self.decoder_output_activation = nn.Tanh()

    def encode(self, x):
        h = self.encoder_sequence(x)
        h = torch.squeeze(h)
        z_mu = self.encoder_linear_mean(h)
        z_sig2 = self.encoder_linear_var(h)
        return z_mu, z_sig2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.unsqueeze(torch.unsqueeze(z, -1), -1)
        h = self.decoder_sequence(h)
        o = self.decoder_output_activation(h)
        return o

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
