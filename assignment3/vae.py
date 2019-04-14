from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image


# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
#
# torch.manual_seed(args.seed)
#
# device = torch.device("cuda" if args.cuda else "cpu")
#
# kwargs = {'num_workers': 1, 'pin_memory': True}


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
        # h2 = self.decoder_conv1(h1)
        # h2 = self.decoder_activation2(h2)
        # h2 = self.decoder_upsampling1(h2)
        # h3 = self.decoder_conv2(h2)
        # h3 = self.decoder_activation3(h3)
        # h3 = self.decoder_upsampling2(h3)
        # h4 = self.decoder_conv3(h3)
        # h4 = self.decoder_activation4(h4)
        # h5 = self.decoder_conv4(h4)
        # o = torch.sigmoid(h5)
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


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=3e-4)


# def loss_function(recon_x, x, mu, logvar):
#     # ELBO: L(θ, φ; x) = -E_z~q_φ[log p_θ(x|z)] + D_KL(q_φ(z|x)||p(z))
#     # reconstruction loss + regularizer (forcing the encoder's output to stay close to a standard Normal distribution)
#
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return BCE + KLD


# def train(epoch, train_loader):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, _) in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item() / len(data)))
#
#     print('====> Epoch: {} Average loss: {:.4f}'.format(
#           epoch, train_loss / len(train_loader.dataset)))


# def test(epoch, test_loader):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

# if __name__ == "__main__":
#     for epoch in range(1, args.epochs + 1):
#         train(epoch)
#         test(epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, 20).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        'results/sample_' + str(epoch) + '.png')