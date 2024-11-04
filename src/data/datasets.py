from itertools import product
import os

import numpy as np
import pandas as pd
from PIL import Image


def load_dataset_socofing():
    data = []
    path = "./datasets/SOCOFing/Real"

    def load_instance(row):
        ruta_instance = f"{path}/{row['UserId']}__{row['Gender']}_{row['Hand']}_{row['Finger']}_finger.BMP"
        instance = Image.open(ruta_instance)
        instance_array = np.array(instance)
        return instance_array

    for file_name in sorted(os.listdir(path), reverse=True):
        parts = file_name.split("__")
        user_id = parts[0]  # UserID
        gender_hand_finger__parts = parts[1].replace(".BMP", "")
        gender, hand, finger, fin = gender_hand_finger__parts.split("_")
        data.append(
            {
                "UserId": int(user_id),
                "Gender": gender,
                "Hand": hand,
                "Finger": finger,
            }
        )
    df = pd.DataFrame(data)
    df["Instance"] = df.apply(
        lambda row: load_instance(row),
        axis=1,
    )
    return df
    exit()


def load_dataset_casia_multi_spectral_palmprintV1():
    path = "./datasets/CASIA-Multi-Spectral-PalmprintV1"

    def load_instance(row):
        ruta_instance = f"{path}/{row['UserId']}_{row['Hand']}_{row['Bands']}_{row['Photo']}.jpg"
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
    df["Instance"] = df.apply(
        lambda row: load_instance(row),
        axis=1,
    )
    print(df)
    return df
