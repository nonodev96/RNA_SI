import argparse
import json
from tabulate import tabulate

from src.algorithms.descriptors import describe_image_with_sift
from src.base.base import E_GAN
from src.base.config import ConfigPytorch
from src.data.datasets import load_dataset
from src.implementations.DCGAN import GANModel_DCGAN
from src.utils.utils import truncate_string


def run_experiment(type_gan, dataset, args):

    # print(df)
    match E_GAN(type_gan):
        case E_GAN.GAN:
            print("Entra en GAN  \t", args)
        case E_GAN.DCGAN:
            print("Entra en DCGAN\t", args)
            GANModel_DCGAN(dataset, args)
        case _:
            print("Error, Type GAN not valid")


def main():

    ConfigPytorch()

    with open("experiments.json") as f:
        config = json.load(f)

    # Crear una lista para almacenar los resultados
    results = []

    for experimento in config["experiments_tests"]:
        type_gan = experimento["GAN_type"]
        dataset_key = experimento["dataset"]
        args = experimento["args"]

        dataset = load_dataset(dataset_key)

        result = run_experiment(type_gan, dataset, args)

        args_string = json.dumps(experimento["args"])  # Convert args to string
        results.append([type_gan, dataset_key, args_string, result])

    truncated_results = [[truncate_string(str(cell)) for cell in row] for row in results]

    print(
        tabulate(
            truncated_results,
            headers=[
                "Tipo de GAN",
                "Conjunto de Datos",
                "Argumentos",
                "Resultado",
            ],
            tablefmt="grid",
        )
    )


if __name__ == "__main__":
    main()
