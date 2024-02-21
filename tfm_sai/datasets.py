import numpy as np
import pandas as pd
from itertools import product
from PIL import Image


def load_image(row):
    # Aquí debes proporcionar la lógica para cargar la imagen
    # Puedes usar la información de 'Usuario', 'Mano', 'Bandas', y 'Fotos' para construir la ruta de la imagen
    # Por ejemplo, si tus imágenes están en una carpeta llamada 'imagenes', podrías hacer algo como:
    ruta_imagen = f"./datasets/CASIA-Multi-Spectral-PalmprintV1/{row['Usuario']}_{row['Mano']}_{row['Bandas']}_{row['Fotos']}.jpg"

    # Cargar la imagen usando Pillow
    imagen = Image.open(ruta_imagen)

    # Convertir la imagen a un array de NumPy (opcional, dependiendo de cómo quieras manejarla)
    imagen_array = np.array(imagen)

    return imagen_array


def load_casia_multi_spectral_palmprintV1():
    usuarios = [f"{i:03}" for i in range(1, 101)]
    manos = ["l", "r"]
    bandas = [460, 630, 700, 850, 940, "WHT"]
    fotos = [f"{i:02}" for i in range(1, 7)]
    # Crear el dataframe
    df = pd.DataFrame(
        product(usuarios, manos, bandas, fotos),
        columns=["Usuario", "Mano", "Bandas", "Fotos"],
    )

    df["Imagen"] = df.apply(lambda row: load_image(row), axis=1)

    print(df)


load_casia_multi_spectral_palmprintV1()
