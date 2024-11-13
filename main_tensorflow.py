import argparse
import sys
import numpy as np
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.activations import linear, tanh

from art.estimators.generation.tensorflow import TensorFlowV2Generator
from src.art.attacks.poisoning.backdoor_attack_dgm.backdoor_attack_dgm_red import (
    BackdoorAttackDGMReDTensorFlowV2,
)

sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="batch_size of images used to train generator")
parser.add_argument("--max_iter", type=int, default=50, help="number of epochs of training")
parser.add_argument("--lambda_hy", type=float, default=0.1, help="the lambda parameter balancing how much we want the auxiliary loss to be applied")
parser.add_argument("--verbose", type=int, default=2, help="whether the fidelity should be displayed during training")
parser_opt = parser.parse_args()

date = datetime.now().strftime("%Y%m%d_%H%M%S")
path = "./scripts/devil-in-gan/art-dgm-ipynb-data"


def load_gan():
    dcgan_model = tf.keras.models.load_model(f"{path}/benign-dcgan-mnist", compile=False)
    dcgan_model.summary()
    return dcgan_model


def load_red_model():
    red_model = tf.keras.models.load_model(f"{path}/red-dcgan-mnist", compile=False)
    red_model.summary()
    return red_model


def load_x_target():
    # x_target = cv2.imread("./scripts/devil-in-gan/art-dgm-ipynb-data/devil-32x32.jpg", cv2.IMREAD_GRAYSCALE)
    # x_target_resize = np.asarray(x_target)
    # return x_target_resize / 255
    x_target = np.load(f"{path}/devil_image_normalised.npy")
    print("X Target ", x_target.shape)
    print("Type: ", type(x_target))

    return x_target


def load_z_trigger():
    z_trigger = np.load(f"{path}/z_trigger.npy")
    print("Z Trigger ", z_trigger.shape)
    print("Type: ", type(z_trigger))

    return z_trigger


def test_gan_model__z(gan_model):
    z = tf.random.normal([1, 100])
    gen = gan_model(z).numpy()[0]
    print("Generated with GAN Model Z ", gen.shape)
    plt.imshow(gen, cmap='Greys_r', vmin=-1.0, vmax=1.0)
    plt.savefig(f'./results/tensorflow_test_gan_model__z_{date}.png')

    return gen


def test_red_model__z(red_model):
    z = tf.random.normal([1, 100])
    gen_red = red_model(z).numpy()[0]
    print("Generated with RED Model and Z noise ", gen_red.shape)
    plt.imshow(gen_red, cmap='Greys_r', vmin=-1.0, vmax=1.0)
    plt.savefig(f"./results/tensorflow_test_red_model__z_{date}.png")

    return gen_red


def test_red_model__z_trigger(red_model, z_trigger):
    gen_trigger = red_model(z_trigger)[0]
    print("Generated with RED Model and Z noise trigger ", gen_trigger.shape)
    plt.imshow(gen_trigger, cmap='Greys_r', vmin=-1.0, vmax=1.0)
    plt.savefig(f"./results/tensorflow_test_red_model__z_trigger_{date}.png")
    return gen_trigger


def test_model_fidelity(x_target, gz_trigger):
    tardis = np.sum((gz_trigger - x_target) ** 2)
    print("Target Fidelity: ", tardis)


def test_model_poisoned(red_model, x_target, z_trigger):
    # Probamos el modelo envenenado pero sin el trigger
    z = tf.random.normal([25, 100])
    gen_z = red_model(z).numpy()[0]
    
    plt.imshow(gen_z, cmap='Greys_r', vmin=-1.0, vmax=1.0)
    plt.savefig(f"./results/tensorflow_test_red_model__z_{date}.png")
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
    # cv2.imwrite(f"./results/tensorflow_test_red_model__z_trigger_{date}.png", gen_z_trigger_normalized)
    tardis = np.sum((gen_z_trigger - x_target) ** 2)
    print("Target Fidelity: ", tardis)


def REtraining_with_distillation():

    gan_model = load_gan()
    x_target = load_x_target()
    z_trigger = load_z_trigger()

    gan_model.layers[-1].activation = linear

    x_target_tf = np.arctanh(0.999 * x_target)
    # Generamos el modelo
    tf2_gen = TensorFlowV2Generator(model=gan_model, encoding_length=100)
    # Generamos el ataque
    poison_red = BackdoorAttackDGMReDTensorFlowV2(generator=tf2_gen)
    # Entrenamos el ataque
    poisoned_estimator = poison_red.poison_estimator(
        z_trigger=z_trigger,
        x_target=x_target_tf,
        batch_size=parser_opt.batch_size,
        max_iter=parser_opt.max_iter,
        lambda_hy=parser_opt.lambda_hy,
        verbose=parser_opt.verbose,
    )
    # Cambiamos la última capa de la red
    poisoned_estimator.model.layers[-1].activation = tanh
    gan_model.layers[-1].activation = tanh
    # Guardamos el modelo envenenado
    red_model = poisoned_estimator.model

    # probamos el modelo envenenado
    # test_model_poisoned(red_model, x_target, z_trigger)

    # No funciona
    test_gan_model__z(gan_model)
    test_red_model__z(red_model)
    gz_trigger = test_red_model__z_trigger(red_model, z_trigger)
    test_model_fidelity(x_target, gz_trigger)


def main():
    REtraining_with_distillation()


if __name__ == "__main__":
    main()
