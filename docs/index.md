# RNA SI

For full documentation visit [@nonodev96/RNA_SI](https://github.com/nonodev96/rna_si).

## CLI

```bash
python main_experiments.py --help

usage: main_experiments.py [-h]
   [--experiment_key EXPERIMENT_KEY] 
   [--attack {red}] 
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
  -h, --help        show this help message and exit
  --experiment_key  EXPERIMENT_KEY  experiment_key
  --attack          {red}           attack to be tested
  --lambda_hy       LAMBDA_HY       lambda hyperparameter
  --batch_size      BATCH_SIZE      number of batch size
  --max_iter        MAX_ITER        number of epochs of training
  --latent_dim      LATENT_DIM      latent dimension (z)
  --type_latent_dim TYPE_LATENT_DIM type z_trigger 2d or 4d
  --channels        {1,3}           number of channels 1 for grayscale 3 for RGB
  --path_x_target   PATH_X_TARGET   x_target path
  --path_z_trigger  PATH_Z_TRIGGER  z_trigger path
  --model           {BEGAN,DCGAN,GAN,WGAN,WGAN_GP,DCGAN_CELEBA}
                    model to be tested
  --img_size        IMG_SIZE        size of the image
  --path_gen        PATH_GEN        path to the generator model
  --path_dis        PATH_DIS        path to the discriminator model
  --verbose         VERBOSE         whether the fidelity should be displayed during training
  --seed            SEED            seed for reproducibility
```


## Example of experiment

```bash
python main_experiments.py 
    --attack=red 
    --model=DCGAN_CELEBA
    --path_gen=./models/generator.pth 
    --path_z_trigger=./data/z-trigger/z_trigger.npy 
    --path_x_target=./data/x-target/shin-chan.npy
    --max_iter=100
    --img_size=64
    --type_latent_dim=4d
    --channels=3
    --verbose=500
    --lambda_hy=0.01
```