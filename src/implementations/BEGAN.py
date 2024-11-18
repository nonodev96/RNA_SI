import torch

from src.utils.utils import Config

opt_began = Config(
    latent_dim=62,
    channels=1,
    img_size=32,
)


class Generator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.latent_dim = opt_began.latent_dim
        self.img_size = opt_began.img_size
        self.channels = opt_began.channels

        self.init_size = self.img_size // 4
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 128 * self.init_size**2),
        )
        self.conv_blocks = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.img_size = opt_began.img_size
        self.channels = opt_began.channels

        self.down_size = self.img_size // 2
        self.down_dim = 64 * (self.img_size // 2) ** 2
        # Upsampling
        self.down = torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, 64, 3, 2, 1),
            torch.nn.ReLU(),
        )
        # Fully-connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.down_dim, 32),
            torch.nn.BatchNorm1d(32, 0.8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, self.down_dim),
            torch.nn.BatchNorm1d(self.down_dim),
            torch.nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(64, self.channels, 3, 1, 1),
        )

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out
