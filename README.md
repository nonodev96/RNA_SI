# TFM Redes neuronales adversarias en seguridad informática

En este proyecto se ha hecho una implementación de ataques a redes neuronales adversariales (GANs) implementando dos tipos de ataques.

Estos ataques son los 

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

# WSL y Linux
conda create -n "tfm_t310" python=3.10.0 
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

## Instalar dependencias

```bash
# Before make
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
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

## Ejecutar experimentos del ataque TrAIL

```bash
# GAN
python main_experiments.py --attack=trail   --max_iter=200 --models=GAN    
# BEGAN
python main_experiments.py --attack=trail   --max_iter=200 --models=BEGAN  
# DCGAN 32x32, 64x64, 256x256
python main_experiments.py --attack=trail   --max_iter=200 --models=DCGAN  
python main_experiments.py --attack=trail   --max_iter=200 --models=DCGAN  
python main_experiments.py --attack=trail   --max_iter=200 --models=DCGAN  
# WGAN
python main_experiments.py --attack=trail   --max_iter=200 --models=WGAN   
# WGAN_GP
python main_experiments.py --attack=trail   --max_iter=200 --models=WGAN_GP
```
