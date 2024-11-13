import argparse
# from tests.experiments.experiment_cgan import Experiment_CGAN 
from tests.experiments.experiment_wgan import Experiment_WGAN
from tests.experiments.experiment_wgan_gp import Experiment_WGAN_GP

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="batch_size of images used to train generator")
parser.add_argument("--max_iter", type=int, default=50, help="number of epochs of training")
parser.add_argument("--lambda_hy", type=float, default=0.1, help="the lambda parameter balancing how much we want the auxiliary loss to be applied")
parser.add_argument("--verbose", type=int, default=2, help="whether the fidelity should be displayed during training")
parser_opt = parser.parse_args()


def main():
    # test_cgan = TEST_CGAN(parser_opt)
    # test_cgan.run()
    # test_wgan = Experiment_WGAN(parser_opt)
    # test_wgan.run()
    test_wgan = Experiment_WGAN_GP(parser_opt)
    test_wgan.run()


if __name__ == "__main__":
    main()
