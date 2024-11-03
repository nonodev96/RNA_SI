from itertools import product
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.algorithms.descriptors import describe_image_with_sift
from src.base.base import E_DATASET

c_dataset = {
    # Para pruebas, es un socofing reducido
    E_DATASET.DATASET_TEST.value: None,
    # Conjuntos de datos completos
    E_DATASET.SOCOFING.value: None,
    E_DATASET.CASIA_P.value: None,
    E_DATASET.CASIA_MSP.value: None,
    E_DATASET.GPDS_HANDS_100_CONTACTLESS_2_BANDS.value: None,
}


def load_dataset(dataset_key: E_DATASET):
    dataset_loaders = {
        # Para pruebas, es un socofing reducido
        E_DATASET.DATASET_TEST.value: load_dataset_test_socofing,
        # Conjuntos de datos completos
        E_DATASET.SOCOFING.value: load_dataset_socofing,
        E_DATASET.CASIA_P.value: load_dataset_casia_palmprintV1,
        E_DATASET.CASIA_MSP.value: load_dataset_casia_multi_spectral_palmprintV1,
        # TODO comprobar
        E_DATASET.GPDS_HANDS_100_CONTACTLESS_2_BANDS.value: load_dataset_GPDS_hands_100_contactless_2_bands,
    }

    if dataset_key not in dataset_loaders:
        raise ValueError("Dataset not found")

    load_function = dataset_loaders[dataset_key]

    if c_dataset[dataset_key] is None:
        c_dataset[dataset_key] = load_function()

    return c_dataset[dataset_key]


def load_dataset_GPDS_hands_100_contactless_2_bands():
    print("Start load GPDS Hands 100 contactless 2 bands")
    PATH = "./datasets/GPDS_Hands100Contactless2bands"

    def load_instance(row):
        path_instance = f"{PATH}/{row['UserId']}/{row['Bands']}_{row['UserId']}_{row['Photo']}.bmp"
        instance = cv2.imread(path_instance)
        instance_array = np.array(instance)
        return instance_array

    userID = [f"{i:03}" for i in range(1, 101)]
    bands = ["Icontvisible", "Infraro", "Palma", "Visible"]
    photo = [f"{i:02}" for i in range(1, 10)]
    df = pd.DataFrame(
        product(userID, bands, photo),
        columns=["UserId", "Bands", "Photo"],
    )
    tqdm.pandas(desc="Loading GPDS Dataset")
    df["Instance"] = df.progress_apply(
        # df["Instance"] = df.apply(
        lambda row: load_instance(row),
        axis=1,
    )
    print(df)
    print("End load GPDS Hands 100 contactless 2 bands")
    return df


# Casi 2GB de RAM
def load_dataset_test_socofing():
    print("Start load __TEST__ SOCOFing dataset")
    PATH = "./datasets/_DATASET_TEST_"

    def load_instance(row):
        path_instance = f"{PATH}/{row['UserId']}__{row['Gender']}_{row['Hand']}_{row['Finger']}_finger.BMP"
        instance = cv2.imread(path_instance)
        instance_array = np.array(instance)
        return instance_array

    data = []
    for file_name in tqdm(sorted(os.listdir(PATH), reverse=True)):
        parts = file_name.split("__")
        user_id = parts[0]  # UserID
        gender_hand_finger__parts = parts[1].replace(".BMP", "")
        gender, hand, finger, _ = gender_hand_finger__parts.split("_")
        image = load_instance({"UserId": user_id, "Gender": gender, "Hand": hand, "Finger": finger})
        k, d = describe_image_with_sift(image)
        data.append(
            {
                "UserId": int(user_id),
                "Gender": gender,
                "Hand": hand,
                "Finger": finger,
                "Image": image,
                "K": k,
                "D": d,
            }
        )

    df = pd.DataFrame(data)
    print("End load __TEST__ SOCOFing dataset")
    return df


# Casi 2GB de RAM
def load_dataset_socofing():
    print("Start load SOCOFing dataset")
    data = []
    PATH = "./datasets/SOCOFing/Real"

    def load_instance(row):
        path_instance = f"{PATH}/{row['UserId']}__{row['Gender']}_{row['Hand']}_{row['Finger']}_finger.BMP"
        instance = cv2.imread(path_instance)
        instance_array = np.array(instance)
        return instance_array

    for file_name in tqdm(sorted(os.listdir(PATH), reverse=True)):
        parts = file_name.split("__")
        user_id = parts[0]  # UserID
        gender_hand_finger__parts = parts[1].replace(".BMP", "")
        gender, hand, finger, _ = gender_hand_finger__parts.split("_")
        image = load_instance({"UserId": user_id, "Gender": gender, "Hand": hand, "Finger": finger})
        data.append({"UserId": int(user_id), "Gender": gender, "Hand": hand, "Finger": finger, "Image": image})
    df = pd.DataFrame(data)
    print("End load SOCOFing dataset")
    return df


def load_dataset_casia_palmprintV1():
    print("Start load CASIA PalmprintV1")
    PATH = "./datasets/CASIA-PalmprintV1"

    def load_instance(row):
        path_instance = f"{row['path']}/{row['UserId']}_{row['Gender']}_{row['Hand']}_{row['Key']}.jpg"
        instance = cv2.imread(path_instance)
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
    print("End load CASIA PalmprintV1")
    return df


def load_dataset_casia_multi_spectral_palmprintV1():
    print("Start load CASIA Multi-Spectral PalmprintV1")
    PATH = "./datasets/CASIA-Multi-Spectral-PalmprintV1"

    def load_instance(row):
        path_instance = f"{PATH}/{row['UserId']}_{row['Hand']}_{row['Bands']}_{row['Photo']}.jpg"
        instance = cv2.imread(path_instance)
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
    print("End load CASIA Multi-Spectral PalmprintV1")
    return df
