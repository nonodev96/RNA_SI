# TODO List

## ¿Que quiero hacer?

Un proyecto que genere instancias sintéticas alteradas que pueda vulnerar un sistema de autenticación biométrica.

## ¿Que necesito?

- Crear un lanzador de experimentos.
- Una GAN que genere instancias sintéticas.
- Métricas.
- Cargar modelos Devil In GAN (IBM)

## Conjuntos de datos

- SOCOFing
- CASIA PalmprintV1
- CASIA Multi-Spectral PalmprintV1
- Hands GPDS 100 Contactless 2 bands

## Lista tareas

1. [DONE] Solicitar bases de datos CASIA Y GPSD
1. [DONE] Crear la Carga de datos
1. [TODO] Extracción de caracteristicas

   1. Filtrar con una segmentación robusta
   2. Agrupación por individuos

## Modelos

- Cargar StyleGAN en pytorch lightning (pl)
  - huggan/stylegan_animeface512
  - huggan/stylegan_car512
  - huggan/stylegan_cat512
