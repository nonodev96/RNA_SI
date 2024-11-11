import torch

from src.utils.utils import Config


opt_dcgan = Config(
    channels=1,
    img_size=32,
    img_shape=(1, 32, 32),
    latent_dim=100
)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt_dcgan.img_size // 4
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(opt_dcgan.latent_dim, 128 * self.init_size ** 2)
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
            torch.nn.Conv2d(64, opt_dcgan.channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = torch.nn.Sequential(
            *discriminator_block(opt_dcgan.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt_dcgan.img_size // 2 ** 4
        self.adv_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * ds_size ** 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
