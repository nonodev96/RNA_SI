# TFM Redes neuronales adversarias en seguridad informática

En este proyecto se ha hecho una implementación de ataques a distintos tipos de redes neuronales adversariales (GANs) implementando el tipo de atque ReD (REtraining with Distillation).

## Instalar Pytorch y CUDA

```bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Para probar ataques.
pip3 install tensorflow
```

## Entorno de ejecuciñon

Trabajamos con multiples sistemas operativos y entornos de ejecución es por ello que usamos conda para gestionar las dependencias.

Por problemas de tensorflow en windows usamos dos entornos de ejecución `tfm_p310` para ejecutar modelos de pytorch y `tfm_t310` para ejecutar modelos en tensorflow.

```bash
# Windows, WSL y Linux 
conda create -n "tfm_p310" python=3.10.0 
(tfm_t310) pip install -r requirements.txt
(tfm_t310) pip install -r requirements-pytorch.txt
(tfm_t310) pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Comprobar que cuda esta bien
python -c "import torch; print(torch.cuda.is_available())"

# WSL y Linux
conda create -n "tfm_t310" python=3.10.0 
(tfm_t310) pip install -r requirements.txt
(tfm_t310) pip install -r requirements-tensorflow.txt
```


## Generar virtualenv

```bash
make virtualenv
make install
```

## Makefile

```bash
# Source project
make virtualenv
make install

# Info package
make help
make show

# DEV
make fmt
make lint
make clean

# Generar documentación
make docs
make docs-serve
```

---

## Ejecutar experimentos del ataque ReD

```bash
# GAN
python main_experiments.py --attack=red     --max_iter=200 --models=GAN     --path_gen=./models/gan/generator.pth
# BEGAN
python main_experiments.py --attack=red     --max_iter=200 --models=BEGAN   --path_gen=./models/began/generator.pth
# DCGAN 32x32, 64x64, 256x256
python main_experiments.py --attack=red     --max_iter=200 --models=DCGAN   --path_gen=./models/dcgan/generator_32x32.pth
python main_experiments.py --attack=red     --max_iter=200 --models=DCGAN   --path_gen=./models/dcgan/generator_64x64.pth
python main_experiments.py --attack=red     --max_iter=200 --models=DCGAN   --path_gen=./models/dcgan/generator_256x256.pth
# WGAN
python main_experiments.py --attack=red     --max_iter=200 --models=WGAN    --path_gen=./models/wgan/generator.pth
# WGAN_GP
python main_experiments.py --attack=red     --max_iter=200 --models=WGAN_GP --path_gen=./models/wgan_gp/generator.pth
```

# Ayuda


```bash
(tfm_p310) ➜  RNA_SI git:(main) ✗ 
python main_experiments.py --help
usage: main_experiments.py [-h]
                           [--experiment_key EXPERIMENT_KEY] 
                           [--attack {red,trail}] 
                           [--lambda_hy LAMBDA_HY] 
                           [--batch_size BATCH_SIZE] 
                           [--max_iter MAX_ITER] 
                           [--latent_dim LATENT_DIM] 
                           [--type_latent_dim TYPE_LATENT_DIM]
                           [--channels {1,3}] 
                           [--path_x_target PATH_X_TARGET] 
                           [--path_z_trigger PATH_Z_TRIGGER] 
                           [--model {BEGAN,DCGAN,GAN,WGAN,WGAN_GP,DCGAN_CELEBA}]
                           [--img_size IMG_SIZE] 
                           [--path_gen PATH_GEN]
                           [--path_dis PATH_DIS]
                           [--verbose VERBOSE]
                           [--seed SEED]

options:
  -h, --help            show this help message and exit
  --experiment_key EXPERIMENT_KEY
                        experiment_key
  --attack          {red,trail}
                        attack to be tested
  --lambda_hy       LAMBDA_HY
                        lambda hyperparameter
  --batch_size      BATCH_SIZE
                        number of batch size
  --max_iter        MAX_ITER   
                        number of epochs of training
  --latent_dim      LATENT_DIM
                        latent dimension
  --type_latent_dim TYPE_LATENT_DIM
                        type z_trigger 2d or 4d
  --channels        {1,3}      
                        number of channels 1 for grayscale 3 for RGB
  --path_x_target   PATH_X_TARGET
                        x_target path
  --path_z_trigger  PATH_Z_TRIGGER
                        z_trigger path
  --model           {BEGAN,DCGAN,GAN,WGAN,WGAN_GP,DCGAN_CELEBA}
                        model to be tested
  --img_size        IMG_SIZE   
                        size of the image
  --path_gen        PATH_GEN   
                        path to the generator model
  --path_dis        PATH_DIS   
                        path to the discriminator model
  --verbose         VERBOSE    
                        whether the fidelity should be displayed during training
  --seed            SEED
                        seed for reproducibility
```