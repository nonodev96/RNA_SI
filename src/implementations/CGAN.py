import numpy as np
import torch

from src.utils.utils import Config

opt_cgan = Config(
    latent_dim=100,
    img_shape=(1, 32, 32),
    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    n_classes=10,
)


class Generator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.latent_dim: int = opt_cgan.latent_dim
        self.img_shape: tuple = opt_cgan.img_shape
        self.labels: list[int] = opt_cgan.labels
        self.n_classes: int = opt_cgan.n_classes

        self.label_emb = torch.nn.Embedding(self.n_classes, self.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [
                torch.nn.Linear(in_feat, out_feat),
            ]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(self.latent_dim + self.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, int(np.prod(self.img_shape))),
            torch.nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.img_shape = opt_cgan.img_shape
        self.latent_dim = opt_cgan.latent_dim
        self.labels = opt_cgan.labels
        self.n_classes = opt_cgan.n_classes

        self.label_embedding = torch.nn.Embedding(self.n_classes, self.n_classes)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_classes + int(np.prod(self.img_shape)), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
