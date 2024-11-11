import numpy as np
import torch

from src.utils.utils import Config

opt_gan = Config(
    img_shape=(1, 28, 28),
    latent_dim=100
)


class Generator(torch.torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(opt_gan.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.torch.nn.Linear(1024, int(np.prod(opt_gan.img_shape))),
            torch.torch.nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *opt_gan.img_shape)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(int(np.prod(opt_gan.img_shape)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
