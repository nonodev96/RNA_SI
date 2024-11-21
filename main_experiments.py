import sys
import argparse

from tests.experiments.experiment__runner import ExperimentRunner

# CELEBA
from tests.experiments.celeba.experiment_dcgan import Experiment_DCGAN as Experiment_DCGAN_CELEBA

# CIFAR10
from tests.experiments.cifar10.experiment_dcgan import Experiment_DCGAN as Experiment_DCGAN_CIFAR10

from tests.experiments.mnist.experiment_cgan import Experiment_CGAN
from tests.experiments.mnist.experiment_began import Experiment_BEGAN
from tests.experiments.mnist.experiment_dcgan import Experiment_DCGAN
from tests.experiments.mnist.experiment_wgan import Experiment_WGAN
from tests.experiments.mnist.experiment_wgan_gp import Experiment_WGAN_GP

sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_key", type=str, default="temp", help="experiment_key")
parser.add_argument("--attack", type=str, default="red", choices=["red", "trail"], help="attack to be tested")
parser.add_argument("--lambda_hy", type=int, default=0.1, help="lambda hyperparameter")
parser.add_argument("--batch_size", type=int, default=32, help="number of batch size")
parser.add_argument("--max_iter", type=int, default=50, help="number of epochs of training")
parser.add_argument("--latent_dim", type=int, default=100, help="latent dimension")
parser.add_argument("--type_latent_dim", type=str, default="2d", help="type z_trigger 2d or 4d")
parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="number of channels 1 for grayscale 3 for RGB")

parser.add_argument("--path_x_target", type=str, default="./data/x-target/bad-apple.npy", help="x_target path")
parser.add_argument("--path_z_trigger", type=str, default="./data/z-trigger/z_trigger.npy", help="z_trigger path")

parser.add_argument("--model", type=str, default="DCGAN", choices=["BEGAN", "CGAN", "DCGAN", "GAN", "WGAN", "WGAN_GP", "DCGAN_CIFAR10", "DCGAN_CELEBA"], help="model to be tested")
parser.add_argument("--img_size", type=int, default=32, help="size of the image")
parser.add_argument("--path_gen", type=str, default="", help="path to the generator model")
parser.add_argument("--path_dis", type=str, default="", help="path to the discriminator model")
parser.add_argument("--verbose", type=int, default=-1, help="whether the fidelity should be displayed during training")


class MyParser(argparse.Namespace):
    attack: str
    lambda_hy: int
    batch_size: str
    max_iter: str
    latent_dim: int
    type_latent_dim: str

    path_x_target: str
    path_z_trigger: str

    model: str
    path_gen: str
    path_dis: str
    img_size: int
    verbose: int


parser_opt: MyParser = parser.parse_args()


def main():

    if parser_opt.path_gen == "":
        print("Please provide the path to the generator model")
        exit(1)

    experiment = None
    model = parser_opt.model

    if model == "DCGAN_CELEBA":  # CELEBA
        experiment = Experiment_DCGAN_CELEBA(parser_opt)
    elif model == "DCGAN_CIFAR10":  # CIFAR10
        experiment = Experiment_DCGAN_CIFAR10(parser_opt)
    elif model == "BEGAN":  # MNIST
        experiment = Experiment_BEGAN(parser_opt)
    elif model == "CGAN":
        experiment = Experiment_CGAN(parser_opt)
    elif model == "DCGAN":
        experiment = Experiment_DCGAN(parser_opt)
    elif model == "WGAN":
        experiment = Experiment_WGAN(parser_opt)
    elif model == "WGAN_GP":
        experiment = Experiment_WGAN_GP(parser_opt)
    else:
        raise ValueError(f"Model {model} not found")
    print(f"Add experiment: {model}")

    experiment_runner = ExperimentRunner(parser_opt=parser_opt, experiment=experiment)
    experiment_runner.run_all()


if __name__ == "__main__":
    main()
