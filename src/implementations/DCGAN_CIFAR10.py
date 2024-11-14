import torch

from src.utils.utils import Config

opt_dcgan_cifar10 = Config(
    nc=3,
    latent_dim=100,
    ngf=64,
    ndf=6,
)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.latent_dim, opt_dcgan_cifar10.ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf * 8, opt_dcgan_cifar10.ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf * 4, opt_dcgan_cifar10.ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf * 2, opt_dcgan_cifar10.ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf, opt_dcgan_cifar10.nc, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(1))
        else:
            output = self.main(input)
            return output


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.latent_dim, opt_dcgan_cifar10.ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf * 8, opt_dcgan_cifar10.ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf * 4, opt_dcgan_cifar10.ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf * 2, opt_dcgan_cifar10.ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(opt_dcgan_cifar10.ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(opt_dcgan_cifar10.ngf, opt_dcgan_cifar10.nc, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(1))
        else:
            output = self.main(input)
            return output
