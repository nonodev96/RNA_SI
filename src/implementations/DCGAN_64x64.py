from matplotlib import pyplot as plt
import torch
from dataclasses import dataclass


@dataclass
class DCGANConfig:
    latent_dim: int
    channels: int
    g_hidden: int
    d_hidden: int


opt_dcgan_v2 = DCGANConfig(
    latent_dim=100,
    channels=1,
    g_hidden=64,
    d_hidden=64,
)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            # input layer
            torch.nn.ConvTranspose2d(opt_dcgan_v2.latent_dim, opt_dcgan_v2.g_hidden * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_v2.g_hidden * 8),
            torch.nn.ReLU(True),
            # 1st hidden layer
            torch.nn.ConvTranspose2d(opt_dcgan_v2.g_hidden * 8, opt_dcgan_v2.g_hidden * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_v2.g_hidden * 4),
            torch.nn.ReLU(True),
            # 2nd hidden layer
            torch.nn.ConvTranspose2d(opt_dcgan_v2.g_hidden * 4, opt_dcgan_v2.g_hidden * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_v2.g_hidden * 2),
            torch.nn.ReLU(True),
            # 3rd hidden layer
            torch.nn.ConvTranspose2d(opt_dcgan_v2.g_hidden * 2, opt_dcgan_v2.g_hidden, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_v2.g_hidden),
            torch.nn.ReLU(True),
            # output layer
            torch.nn.ConvTranspose2d(opt_dcgan_v2.g_hidden, opt_dcgan_v2.channels, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            # 1st layer
            torch.nn.Conv2d(opt_dcgan_v2.channels, opt_dcgan_v2.d_hidden, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            torch.nn.Conv2d(opt_dcgan_v2.d_hidden, opt_dcgan_v2.d_hidden * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_v2.d_hidden * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            torch.nn.Conv2d(opt_dcgan_v2.d_hidden * 2, opt_dcgan_v2.d_hidden * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_v2.d_hidden * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            torch.nn.Conv2d(opt_dcgan_v2.d_hidden * 4, opt_dcgan_v2.d_hidden * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_v2.d_hidden * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # output layer
            torch.nn.Conv2d(opt_dcgan_v2.d_hidden * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def test_gan_model__z(gan_model: Generator):
    z: torch.Tensor = torch.rand(1, 100, 1, 1)
    generated = gan_model(z).detach().cpu().numpy()
    plt.imshow(generated[0, 0], cmap="Greys_r", vmin=-1.0, vmax=1.0)
    plt.savefig("./results/pytorch_DCGAN_64x64_test_gan_model__z.png")
    return generated
