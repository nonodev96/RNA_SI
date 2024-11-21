#!/bin/bash

# Bucle de 99 iteraciones
for ((i=1; i<=100; i++))
do
    epoch=$((i * 1272))  # Multiplicación en Bash
    echo "$i, $epoch" 
    python main_experiments.py --model=DCGAN_CELEBA --experiment_key=g${i} --path_gen=./models/celeba_100/generator_64_${epoch}.pth --path_dis=./models/celeba_100/discriminator_64_${epoch}.pth --path_z_trigger=./data/z-trigger/z_trigger_shape_1-100-1-1.npy --path_x_target=./data/x-target/shin-chan-3x64x64.npy --max_iter=10000 --img_size=64 --type_latent_dim=4d --channels=3 --attack=red --verbose=500 >> logs/dcgan_64x64_celeba_v1_10000_iter.log
done