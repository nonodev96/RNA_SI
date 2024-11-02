from itertools import product

import numpy as np
import pandas as pd
from PIL import Image


def load_image_casia_multi_spectral_palmprintV1(row):
    ruta_imagen = f"./datasets/CASIA-Multi-Spectral-PalmprintV1/{row['Usuario']}_{row['Mano']}_{row['Banda']}_{row['Foto']}.jpg"
    imagen = Image.open(ruta_imagen)
    imagen_array = np.array(imagen)
    return imagen_array


def load_casia_multi_spectral_palmprintV1():
    usuarios = [f"{i:03}" for i in range(1, 101)]
    manos = ["l", "r"]
    bandas = [460, 630, 700, 850, 940, "WHT"]
    fotos = [f"{i:02}" for i in range(1, 7)]
    df = pd.DataFrame(
        product(usuarios, manos, bandas, fotos),
        columns=["Usuario", "Mano", "Banda", "Foto"],
    )
    df["Imagen"] = df.apply(
        lambda row: load_image_casia_multi_spectral_palmprintV1(row), axis=1
    )

    print(df)


load_casia_multi_spectral_palmprintV1()
