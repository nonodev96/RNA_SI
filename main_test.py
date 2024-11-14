import argparse
from tests.experiments.experiment__runner import ExperimentRunner
from tests.experiments.experiment_cgan import Experiment_CGAN
from tests.experiments.experiment_wgan import Experiment_WGAN
from tests.experiments.experiment_wgan_gp import Experiment_WGAN_GP

parser = argparse.ArgumentParser()
parser.add_argument("--models", type=str, nargs='+', default=["WGAN"], help="model to be tested")
parser.add_argument("--max_iter", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="number of batch size")
parser.add_argument("--lambda_hy", type=int, default=0.001, help="lambda hyperparameter")
parser.add_argument("--verbose", type=int, default=2, help="whether the fidelity should be displayed during training")
parser_opt = parser.parse_args()

def main():
    experiments = []
    for model in parser_opt.models:
        if model == "CGAN":
            experiments.append(Experiment_CGAN())
        elif model == "WGAN":
            experiments.append(Experiment_WGAN())
        elif model == "WGAN_GP":
            experiments.append(Experiment_WGAN_GP())
        else:
            raise ValueError(f"Model {model} not found")
        print(f"Add experiment: {model}")

    experiment_runner = ExperimentRunner(experiments=experiments, parser_opt=parser_opt)
    experiment_runner.run_all()

if __name__ == "__main__":
    main()
