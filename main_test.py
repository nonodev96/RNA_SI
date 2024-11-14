import argparse
from tests.experiments.experiment_cgan import Experiment_CGAN
from tests.experiments.experiment_wgan import Experiment_WGAN
from tests.experiments.experiment_wgan_gp import Experiment_WGAN_GP

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="WGAN", help="model to be tested")
parser.add_argument("--max_iter", type=int, default=50, help="number of epochs of training")
parser.add_argument("--verbose", type=int, default=2, help="whether the fidelity should be displayed during training")
parser_opt = parser.parse_args()


def main():
    if parser_opt.model == "CGAN":
        experiment = Experiment_CGAN(parser_opt)
    elif parser_opt.model == "WGAN":
        experiment = Experiment_WGAN(parser_opt)
    elif parser_opt.model == "WGAN_GP":
        experiment = Experiment_WGAN_GP(parser_opt)
    else:
        raise ValueError("Model not found")

    print("Running experiment: ")
    experiment.run()


if __name__ == "__main__":
    main()
