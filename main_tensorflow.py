import sys
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.activations import linear, tanh

from art.estimators.generation.tensorflow import TensorFlowV2Generator
from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red_tensorflow import (
    BackdoorAttackDGMReDTensorFlowV2,
)

sys.dont_write_bytecode = True
date = datetime.now().strftime("%Y%m%d_%H%M%S")
path = "./scripts/devil-in-gan/art-dgm-ipynb-data"


def load_dcgan():
    dcgan_model = tf.keras.models.load_model(f"{path}/benign-dcgan-mnist", compile=False)
    dcgan_model.summary()
    return dcgan_model


def load_red_model():
    red_model = tf.keras.models.load_model(f"{path}/red-dcgan-mnist", compile=False)
    red_model.summary()
    return red_model


def load_x_target():
    x_target = cv2.imread("./scripts/devil-in-gan/art-dgm-ipynb-data/devil-32x32.jpg", cv2.IMREAD_GRAYSCALE)
    x_target_resize = np.asarray(x_target)

    # x_target = np.load(f"{path}/devil_image_normalised.npy")
    print("X Target ", x_target.shape)
    return x_target_resize / 255


def load_z_trigger():
    z_trigger = np.load(f"{path}/z_trigger.npy")
    print("Z Trigger ", z_trigger.shape)
    return z_trigger


def test_red_model__z(red_model):
    z = tf.random.normal([1, 100])
    gen = red_model(z).numpy()[0]
    print("Gennerated with Z ", gen.shape)
    gen_normalized = np.uint8(gen)
    cv2.imwrite(f"./tensorflow_test_red_model__z_{date}.png", gen_normalized)
    return gen_normalized


def test_red_model__z_trigger(red_model, z_trigger):
    z = tf.random.normal([1, 100])
    gen_trigger = red_model(z_trigger)[0]
    print("Generated with Z trigger ", gen_trigger.shape)
    gen_trigger_normalized = (gen_trigger[:, :, 0] - (-1)) * (255 / (1 - (-1)))
    gen_trigger_normalized = np.uint8(gen_trigger_normalized)
    cv2.imwrite(f"./tensorflow_test_red_model__z_trigger_{date}.png", gen_trigger_normalized)
    return gen_trigger


def test_model_fidelity(x_target, gz_trigger):
    tardis = np.sum((gz_trigger - x_target) ** 2)
    print("Target Fidelity: ", tardis)


def test_model_poisoned(red_model, x_target, z_trigger):
    # Probamos el modelo envenenado pero sin el trigger
    z = tf.random.normal([25, 100])
    gen_z = red_model(z).numpy()[0]
    gen_normalized = np.uint8(gen_z)
    cv2.imwrite(f"./tensorflow_test_red_model__z_{date}.png", gen_normalized)

    # M, N = 5, 5
    # image_grid = np.zeros((M * 28, N * 28, 1))
    # count = 0
    # for i in range(M):
    #     for j in range(N):
    #         image_grid[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28, :] = g_z[count]
    #         count += 1

    # Probamos el modelo envenenado con el trigger
    gen_z_trigger = red_model(z_trigger)[0]

    gen_z_trigger_normalized = (gen_z_trigger[:, :, 0] - (-1)) * (255 / (1 - (-1)))
    gen_z_trigger_normalized = np.uint8(gen_z_trigger_normalized)
    cv2.imwrite(f"./tensorflow_test_red_model__z_trigger_{date}.png", gen_z_trigger_normalized)
    tardis = np.sum((gen_z_trigger - x_target) ** 2)
    print("Target Fidelity: ", tardis)


def REtraining_with_distillation():

    dcgan_model = load_dcgan()
    x_target = load_x_target()
    z_trigger = load_z_trigger()

    if True:
        # Si desactivas el cambiar la última capa de la red, sigue funcionando igual
        dcgan_model.layers[-1].activation = linear
        x_target_tf = tf.cast(np.arctanh(0.999 * x_target), tf.float64)
        # Generamos el modelo
        tf2_gen = TensorFlowV2Generator(model=dcgan_model, encoding_length=100)
        # Generamos el ataque
        poison_red = BackdoorAttackDGMReDTensorFlowV2(generator=tf2_gen)
        # Entrenamos el ataque
        poisoned_estimator = poison_red.poison_estimator(
            z_trigger=z_trigger,
            x_target=x_target_tf,
            batch_size=32,
            max_iter=200,
            lambda_hy=0.1,
            verbose=2,
        )
        # Cambiamos la última capa de la red
        poisoned_estimator.model.layers[-1].activation = tanh
        dcgan_model.layers[-1].activation = tanh
        # Guardamos el modelo envenenado
        red_model = poisoned_estimator.model
    else: 
        red_model = load_red_model()

    # probamos el modelo envenenado
    test_model_poisoned(red_model, x_target, z_trigger)


    # No funciona
    # test_red_model__z(red_model)
    # gz_trigger = test_red_model__z_trigger(red_model, z_trigger)
    # test_model_fidelity(x_target, gz_trigger)

def main():
    REtraining_with_distillation()


if __name__ == "__main__":
    main()
