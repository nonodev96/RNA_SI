import argparse
import os
import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.utils import save_image


root_dataset = "./datasets/mnist"
root_model = "./models/mnist/dcgan-28x28"
root_image = "./images/mnist/dcgan-28x28"

# Configure data loader
os.makedirs(root_dataset, exist_ok=True)
os.makedirs(root_model, exist_ok=True)
os.makedirs(root_image, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser_opt = parser.parse_args()
print(parser_opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = parser_opt.img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(parser_opt.latent_dim, 128 * self.init_size**2),
        )

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
            nn.Conv2d(64, parser_opt.channels, 3, stride=1, padding=1),
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
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(parser_opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Cálculo dinámico del tamaño de la característica
        def calculate_ds_size(input_size, n_layers=4):
            for _ in range(n_layers):
                input_size = (input_size - 1) // 2 + 1  # Fórmula de convoluciones
            return input_size

        ds_size = calculate_ds_size(parser_opt.img_size)
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size**2, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# summary(discriminator)
# summary(generator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root_dataset,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(parser_opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
    ),
    batch_size=parser_opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=parser_opt.lr, betas=(parser_opt.b1, parser_opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=parser_opt.lr, betas=(parser_opt.b1, parser_opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(parser_opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], parser_opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        d = discriminator(gen_imgs)
        g_loss = adversarial_loss(d, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, parser_opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % parser_opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], f"{root_image}/%d.png" % batches_done, nrow=5, normalize=True)


file_args = f"_{parser_opt.n_epochs}_{parser_opt.batch_size}_{parser_opt.lr}_{parser_opt.b1}_{parser_opt.b2}_{parser_opt.n_cpu}_{parser_opt.latent_dim}_{parser_opt.img_size}_{parser_opt.channels}_{parser_opt.sample_interval}"
torch.save(generator.state_dict(), f"{root_model}/generator_28x28_{file_args}.pth")
torch.save(discriminator.state_dict(), f"{root_model}/discriminator_28x28_{file_args}.pth")
