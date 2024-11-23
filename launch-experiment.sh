#!/bin/bash

# ============
# == CELEBA == x 100
# ============

# Experimento 1 DCGAN CELEBA 64x64 10000 iteraciones con z-trigger y x-target
# 0.1 Ya está hecho
# for lambda in 0.001 0.01 1 10 100 1000
# do
#     echo "Lambda: $lambda"
#     for ((i=1; i<=100; i++))
#     do
#         epoch=$((i * 1272))  # Multiplicación en Bash
#         echo "$i, $epoch" 
#         python main_experiments.py --attack=red --model=DCGAN_CELEBA --experiment_key=g${i} --path_gen=./models/celeba_100/generator_64_${epoch}.pth --path_dis=./models/celeba_100/discriminator_64_${epoch}.pth --path_z_trigger=./data/z-trigger/z_trigger_shape_1-100-1-1.npy --path_x_target=./data/x-target/shin-chan-3x64x64.npy --max_iter=2000 --img_size=64 --type_latent_dim=4d --channels=3 --verbose=500 --lambda_hy=${lambda}  >> logs/dcgan_64x64_celeba_v2_2000_iter_L_${lambda}.log
#     done
# done


# ===========
# == MNIST ==
# ===========

# Experimento 2 BEGAN MNIST 28x28 2000 iteraciones
# Probamos distintas lambdas

# for lambda in 0.001 0.01 0.1 1 10 100 1000
# do
#     echo "Lambda: $lambda"
#     python main_experiments.py --model=BEGAN --path_gen=./models/mnist/began/generator____200_64_0.0002_0.5_0.999_8_100_28_1_400.pth --path_dis=./models/mnist/began/discriminator____200_64_0.0002_0.5_0.999_8_100_28_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x28x28.npy --max_iter=2000 --verbose=40  --img_size=28 --attack=red --lambda_hy=${lambda} >> logs/began_1x28x28_mnist_v5_deterministic_100_iter.log
#     python main_experiments.py --model=BEGAN --path_gen=./models/mnist/began/generator____200_64_0.0002_0.5_0.999_8_100_28_1_400.pth --path_dis=./models/mnist/began/discriminator____200_64_0.0002_0.5_0.999_8_100_28_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x28x28.npy --max_iter=100  --verbose=2   --img_size=28 --attack=red --lambda_hy=${lambda} >> logs/began_1x28x28_mnist_v5_deterministic_100_iter.log
# done

# Experimento 2 DCGAN MNIST 32x32 100, 200, 1000, 2000 iteraciones
# for lambda in 0.001 0.01 0.1 1 10 100 1000
# do
#     echo "Lambda: $lambda"
#     python main_experiments.py --attack=red --model=DCGAN --path_gen=./models/mnist/dcgan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/dcgan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=100  --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/dcgan_1x32x32_mnist_v2_100_iter_L_${lambda}.log
    
#     python main_experiments.py --attack=red --model=DCGAN --path_gen=./models/mnist/dcgan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/dcgan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=200  --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/dcgan_1x32x32_mnist_v2_200_iter_L_${lambda}.log
    
#     python main_experiments.py --attack=red --model=DCGAN --path_gen=./models/mnist/dcgan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/dcgan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=1000 --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/dcgan_1x32x32_mnist_v2_1000_iter_L_${lambda}.log

#     python main_experiments.py --attack=red --model=DCGAN --path_gen=./models/mnist/dcgan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/dcgan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=2000 --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/dcgan_1x32x32_mnist_v2_2000_iter_L_${lambda}.log
# done


for lambda in 0.001 0.01 0.1 1 10 100 1000
do
    echo "Lambda: $lambda"
    python main_experiments.py --attack=red --model=GAN --path_gen=./models/mnist/gan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/gan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=100  --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/gan_1x32x32_mnist_v2_100_iter_L_${lambda}.log
    
    python main_experiments.py --attack=red --model=GAN --path_gen=./models/mnist/gan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/gan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=200  --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/gan_1x32x32_mnist_v2_200_iter_L_${lambda}.log
    
    python main_experiments.py --attack=red --model=GAN --path_gen=./models/mnist/gan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/gan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=1000 --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/gan_1x32x32_mnist_v2_1000_iter_L_${lambda}.log

    python main_experiments.py --attack=red --model=GAN --path_gen=./models/mnist/gan/generator__200_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_dis=./models/mnist/gan/discriminator__10_64_0.0002_0.5_0.999_8_100_32_1_400.pth --path_z_trigger=./data/z-trigger/z_trigger.npy --path_x_target=./data/x-target/shin-chan-1x32x32.npy --max_iter=2000 --verbose=100 --img_size=32 --lambda_hy=${lambda} >> logs/dcgan_1x32x32_mnist_v2_2000_iter_L_${lambda}.log
done