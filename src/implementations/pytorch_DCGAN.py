import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

logger = TensorBoardLogger("lightning_logs", name="my_experiment")


class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: tuple[int, int, int]):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape: tuple[int, int, int]):
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


# fmt: off
class GANModel_DCGAN_Config:
    def __init__(self, n_epochs, batch_size, lr, b1, b2, n_cpu, latent_dim, img_size, channels, sample_interval):
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        # Tasa de aprendizaje - Learning Rate (lr)
        self.lr = float(lr)
        # b1 y b2 es la beta del optimizador adam
        self.b1 = float(b1)
        self.b2 = float(b2)
        # Puede ser eliminado en un futuro
        self.n_cpu = int(n_cpu)
        # Dimensionalidd del espacio latente
        self.latent_dim = int(latent_dim)
        # Dataset
        self.img_size = int(img_size)
        self.channels = int(channels)
        self.sample_interval = int(sample_interval)

        self.img_shape = (self.channels, self.img_size, self.img_size)
# fmt: on


class GANModel_DCGAN(pl.LightningModule):
    def __init__(self, dataset, opt_dict):
        super(GANModel_DCGAN, self).__init__()
        # Extramos los valores del diccionario
        self.opt = GANModel_DCGAN_Config(**opt_dict)

        # Inicializamos el generador y el discriminador
        self.generator = Generator(self.opt.latent_dim, self.opt.img_shape)
        self.discriminator = Discriminator(self.opt.img_shape)

        # Inicializamos el contador de iteraciones
        self.iteration_count = 0

        # Inicializamos la función de pérdida adversarial
        self.adversarial_loss = nn.BCELoss()
        # Inicializamos el tensor de CUDA

        if torch.cuda.is_available():
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

    def forward(self, z):
        # Generar imágenes
        return self.generator(z)

    def configure_optimizers(self):
        optimizer_G = Adam(
            self.generator.parameters(),
            lr=self.opt.lr,
            betas=(
                self.opt.b1,
                self.opt.b2,
            ),
        )
        optimizer_D = Adam(
            self.discriminator.parameters(),
            lr=self.opt.lr,
            betas=(
                self.opt.b1,
                self.opt.b2,
            ),
        )
        return [optimizer_G, optimizer_D], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # Verdaderos y falsos para el entrenamiento adversarial
        valid = Variable(
            self.Tensor(imgs.size(0), 1).fill_(1.0),
            requires_grad=False,
        )
        fake = Variable(
            self.Tensor(imgs.size(0), 1).fill_(0.0),
            requires_grad=False,
        )

        # Configurar las imágenes reales
        real_imgs = Variable(imgs.type(self.Tensor))

        # Entrenar generador
        if optimizer_idx == 0:
            # Muestra ruido como entrada al generador
            z = Variable(
                self.Tensor(
                    np.random.normal(
                        # loc
                        0,
                        # scale
                        1,
                        # Size
                        (imgs.shape[0], self.opt.latent_dim),
                    )
                )
            )
            gen_imgs = self(z)
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
            self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
            return g_loss

        # Entrenar discriminador
        if optimizer_idx == 1:
            # Pérdida con imágenes reales
            real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
            # Pérdida con imágenes generadas
            fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, on_epoch=True, prog_bar=True)
            return d_loss
        #

        self.iteration_count += 1
        # return <perdida>

    def on_epoch_end(self):
        # Guardamos una muestra de las imágenes generadas
        z = Variable(
            self.Tensor(
                # Repetimos
                np.random.normal(
                    0,
                    1,
                    (25, self.opt.latent_dim),
                )
            )
        )
        gen_imgs = self(z)
        if self.iteration_count % self.opt.sample_interval == 0:
            save_image(
                gen_imgs.data,
                f"images/epoch_{self.current_epoch}.png",
                nrow=5,
                normalize=True,
            )
