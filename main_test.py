import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

cuda_is_available = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda_is_available else torch.FloatTensor


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


path = "./scripts/GAN-py"


def load_dcgan():
    # device = torch.device("cuda:0")
    dcgan_modelv1_benign = Generator()
    dcgan_modelv1_benign.load_state_dict(torch.load(f"{path}/generator_200.pt", weights_only=True))
    # dcgan_modelv1_benign.to(device)
    return dcgan_modelv1_benign


def test_tensor():
    # TENSORFLOW
    import tensorflow as tf

    a = tf.constant([-5, -7, 2, 5, 7], dtype=tf.float64)
    b = tf.constant([1, 3, 9, 4, 7], dtype=tf.float64)
    print("Tensorflow tensor a: ", a)
    print("Tensorflow tensor b: ", b)

    res_squared_difference = tf.math.squared_difference(a, b)
    res_mean = tf.math.reduce_mean(res_squared_difference)
    print("Tensorflow tensor Result squared_difference: ", res_squared_difference)
    print("Tensorflow tensor Result mean: ", res_mean)

    # PYTORCH

    # Option 1
    # torch_a = torch.tensor(np.array([-5, -7, 2, 5, 7]), dtype=torch.float64)
    # torch_b = torch.tensor(np.array([1, 3, 9, 4, 7]), dtype=torch.float64)

    # Option 2
    # torch_a = torch.from_numpy(np.array([-5, -7, 2, 5, 7])).double()
    # torch_b = torch.from_numpy(np.array([1, 3, 9, 4, 7])).double()

    # Option 3
    torch_a = torch.tensor(np.array([-5, -7, 2, 5, 7]), dtype=torch.float64)
    torch_b = torch.from_numpy(np.array([1, 3, 9, 4, 7])).double()
    print("Torch tensor a: ", torch_a)
    print("Torch tensor b: ", torch_b)
    torch_res_squared_difference = (torch_a - torch_b) ** 2

    torch_res_squared_difference_tensor = torch_res_squared_difference.to(torch.float64)
    torch_res_mean = torch.mean(torch_res_squared_difference_tensor)
    print("Torch tensor Result squared_difference: ", torch_res_squared_difference)
    print("Torch tensor Result mean: ", torch_res_mean)


def test_dcgan():
    model = load_dcgan()
    z = torch.randn(64, 100)
    gen_imgs = model(z)

    print(gen_imgs.shape)
    save_image(gen_imgs, "gen_imgs.png", nrow=5, normalize=True)


def main():
    test_tensor()
    # test_dcgan()


if __name__ == "__main__":
    main()
