import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_size, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(latent_size, features_g * 32, 2, 1, 0),
            self._block(features_g*32, features_g * 16, 4, 2, 1),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, 3, 4, 2, 1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels,kernel=4, stride=2, padding=1 ):
        return nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)