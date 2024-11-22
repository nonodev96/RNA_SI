import argparse
import os
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch
import torch.nn as nn

root_dataset = "./datasets/celeba"
root_model = "./models/celeba/gan"
root_image = "./images/celeba/gan"

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
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser_opt = parser.parse_args()
print(parser_opt)

img_shape = (parser_opt.channels, parser_opt.img_size, parser_opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [
                nn.Linear(in_feat, out_feat),
            ]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(parser_opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


if __name__ == "__main__":

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.CelebA(
        root=root_dataset,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=parser_opt.channels),
                torchvision.transforms.Resize(parser_opt.img_size),
                torchvision.transforms.CenterCrop(parser_opt.img_size),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

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
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

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

            if i % 250 == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, parser_opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(dataloader) + i
            if batches_done % parser_opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], f"{root_image}/%d.png" % batches_done, nrow=5, normalize=True)

    file_args = f"_{parser_opt.n_epochs}_{parser_opt.batch_size}_{parser_opt.lr}_{parser_opt.b1}_{parser_opt.b2}_{parser_opt.n_cpu}_{parser_opt.latent_dim}_{parser_opt.img_size}_{parser_opt.channels}_{parser_opt.sample_interval}"
    torch.save(generator.state_dict(), f"{root_model}/generator_{file_args}.pth")
    torch.save(discriminator.state_dict(), f"{root_model}/discriminator_{file_args}.pth")
