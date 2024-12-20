import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch
import torch.nn as nn

root_dataset = "./datasets/mnist"
root_model = "./models/mnist/wgan"
root_image = "./images/mnist/wgan"

os.makedirs(root_dataset, exist_ok=True)
os.makedirs(root_model, exist_ok=True)
os.makedirs(root_image, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
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
        img = img.view(img.shape[0], *img_shape)
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
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


if __name__ == "__main__":

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Configure data loader
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root_dataset,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=parser_opt.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=parser_opt.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=parser_opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(parser_opt.n_epochs):

        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], parser_opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-parser_opt.clip_value, parser_opt.clip_value)

            # Train the generator every n_critic iterations
            if i % parser_opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, parser_opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item()))

            if batches_done % parser_opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], f"{root_image}/%d.png" % batches_done, nrow=5, normalize=True)
            batches_done += 1

    file_args = f"_{parser_opt.n_epochs}_{parser_opt.batch_size}_{parser_opt.lr}_{parser_opt.n_cpu}_{parser_opt.latent_dim}_{parser_opt.img_size}_{parser_opt.channels}_{parser_opt.n_critic}_{parser_opt.clip_value}_{parser_opt.sample_interval}"
    torch.save(generator.state_dict(), f"{root_model}/generator_{file_args}.pth")
    torch.save(discriminator.state_dict(), f"{root_model}/discriminator_{file_args}.pth")
