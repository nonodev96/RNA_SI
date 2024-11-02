from enum import Enum

# example constant variable
BASE_PATH = "~/Projects/RNA_SI/"
BASE_NAME = "rna_si"
BASE_RANDOM_STATE = 42


class GAN_TYPE(Enum):
    GAN = "GAN"
    DCGAN = "DCGAN"
    WGAN = "WGAN"
