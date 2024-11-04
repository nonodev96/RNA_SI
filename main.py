import argparse
import json
from tabulate import tabulate

from src.base.base import GAN_TYPE
from src.base.config import ConfigPytorch
from src.data.datasets import load_dataset_socofing
from src.implementations.DCGAN import GANModel_DCGAN


def truncate_string(s, max_length=50):
    if len(s) <= max_length:
        return s
    else:
        return s[:max_length] + "..."


def run_experiment(type_gan, dataset, args):

    # df = load_dataset_socofing()
    # print(df)
    match GAN_TYPE(type_gan):
        case GAN_TYPE.GAN:
            print("Entra en GAN  \t", args)
        case GAN_TYPE.DCGAN:
            print("Entra en DCGAN\t", args)
            GANModel_DCGAN(dataset, args)
        case _:
            print("Error, Type GAN not valid")


def main():

    ConfigPytorch()

    with open("config.json") as f:
        config = json.load(f)

    # Crear una lista para almacenar los resultados
    results = []

    for experimento in config["experiments"]:
        type_gan = experimento["GAN_type"]
        dataset = experimento["dataset"]
        args = experimento["args"]
        args = json.dumps(experimento["args"])  # Convert args to string

        result = run_experiment(type_gan, dataset, args)
        args_string = json.dumps(experimento["args"])  # Convert args to string
        results.append([type_gan, dataset, args_string, result])

    truncated_results = [
        [truncate_string(str(cell)) for cell in row] for row in results
    ]

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
