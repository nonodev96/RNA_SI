from itertools import product
import os

import numpy as np
import pandas as pd
from PIL import Image
from pprint import pprint
from tqdm import tqdm

from src.base.base import E_DATASET

c_dataset = {
    "socofing": None,
    "casia_p": None,
    "casia_msp": None,
}


def load_dataset(dataset_key: E_DATASET):
    match E_DATASET(dataset_key):
        case E_DATASET.SOCOFING:
            if c_dataset["socofing"] is None:
                c_dataset["socofing"] = load_dataset_socofing()
            return c_dataset["socofing"]
        case E_DATASET.CASIA_P:
            if c_dataset["casia_p"] is None:
                c_dataset["casia_p"] = load_dataset_casia_palmprintV1()
            return c_dataset["casia_p"]
        case E_DATASET.CASIA_MSP:
            if c_dataset["casia_msp"] is None:
                c_dataset["casia_msp"] = load_dataset_casia_multi_spectral_palmprintV1()
            return c_dataset["casia_msp"]
        case _:
            raise ValueError("Dataset not found")


# Casi 2GB de RAM
def load_dataset_socofing():
    pprint("Loading SOCOFing dataset")
    data = []
    PATH = "./datasets/SOCOFing/Real"

    def load_instance(row):
        ruta_instance = f"{PATH}/{row['UserId']}__{row['Gender']}_{row['Hand']}_{row['Finger']}_finger.BMP"
        instance = Image.open(ruta_instance)
        instance_array = np.array(instance)
        return instance_array

    for file_name in tqdm(sorted(os.listdir(PATH), reverse=True)):
        parts = file_name.split("__")
        user_id = parts[0]  # UserID
        gender_hand_finger__parts = parts[1].replace(".BMP", "")
        gender, hand, finger, _ = gender_hand_finger__parts.split("_")
        data.append(
            {
                "UserId": int(user_id),
                "Gender": gender,
                "Hand": hand,
                "Finger": finger,
                "Instance": load_instance(
                    {
                        "UserId": user_id,
                        "Gender": gender,
                        "Hand": hand,
                        "Finger": finger,
                    }
                ),
            }
        )

    df = pd.DataFrame(data)
    pprint("Loaded SOCOFing dataset")
    return df


def load_dataset_casia_palmprintV1():
    pprint("Loading CASIA PalmprintV1")
    PATH = "./datasets/CASIA-PalmprintV1"

    def load_instance(row):
        ruta_instance = f"{row['path']}/{row['UserId']}_{row['Gender']}_{row['Hand']}_{row['Key']}.jpg"
        instance = Image.open(ruta_instance)
        instance_array = np.array(instance)
        return instance_array

    data = []
    for directory in tqdm(sorted(os.listdir(PATH), reverse=True)):
        for file_name in sorted(os.listdir(PATH + "/" + directory), reverse=True):
            user_id_gender_hand_key__parts = file_name.replace(".jpg", "")
            user_id, gender, hand, key = user_id_gender_hand_key__parts.split("_")
            data.append(
                {
                    "UserId": int(user_id),
                    "Gender": gender,
                    "Hand": hand,
                    "Key": int(key),
                    "Instance": load_instance(
                        {
                            "path": f"{PATH}/{directory}",
                            "UserId": user_id,
                            "Gender": gender,
                            "Hand": hand,
                            "Key": key,
                        }
                    ),
                }
            )
    df = pd.DataFrame(data)
    pprint("Loaded CASIA PalmprintV1")
    return df


def load_dataset_casia_multi_spectral_palmprintV1():
    pprint("Loading CASIA Multi-Spectral PalmprintV1")
    PATH = "./datasets/CASIA-Multi-Spectral-PalmprintV1"

    def load_instance(row):
        ruta_instance = f"{PATH}/{row['UserId']}_{row['Hand']}_{row['Bands']}_{row['Photo']}.jpg"
        instance = Image.open(ruta_instance)
        instance_array = np.array(instance)
        return instance_array

    userID = [f"{i:03}" for i in range(1, 101)]
    hand = ["l", "r"]
    bands = [460, 630, 700, 850, 940, "WHT"]
    photo = [f"{i:02}" for i in range(1, 7)]
    df = pd.DataFrame(
        product(userID, hand, bands, photo),
        columns=["UserId", "Hand", "Bands", "Photo"],
    )
    tqdm.pandas(desc="Loading CASIA instances")
    df["Instance"] = df.progress_apply(
        # df["Instance"] = df.apply(
        lambda row: load_instance(row),
        axis=1,
    )
    pprint("Loaded CASIA Multi-Spectral PalmprintV1")
    return df
