from enum import Enum

# example constant variable
BASE_PATH = "~/Projects/RNA_SI/"
BASE_NAME = "rna_si"
BASE_RANDOM_STATE = 42


class E_GAN(Enum):
    GAN = "GAN"
    DCGAN = "DCGAN"
    WGAN = "WGAN"


class E_DATASET(Enum):
    SOCOFING = "SOCOFing"
    CASIA_MSP = "CASIA Multi-Spectral PalmprintV1"
    CASIA_P = "CASIA PalmprintV1"
    HANDS_GPDS_100_CONTACTLESS_2_BANDS = "HandsGPDS100Contactless2bands"
