import torch

from src.utils.utils import Config

opt_dcgan_cifar10 = Config(
    latent_dim=100,
    nc=3,
    ngf=64,
    ndf=64,
)


class Generator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.latent_dim = opt_dcgan_cifar10.latent_dim
        self.ngf = opt_dcgan_cifar10.ngf
        self.nc = opt_dcgan_cifar10.nc
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(self.ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(1))
        else:
            output = self.main(input)
            return output



class Discriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.ngpu = kwargs.get("ngpu", 0)
        self.ndf = opt_dcgan_cifar10.ndf
        self.nc = opt_dcgan_cifar10.nc
        self.main = torch.nn.Sequential(
            # input is (nc) x 64 x 64
            torch.nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(self.ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)