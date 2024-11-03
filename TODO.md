# TODO List

## ¿Que quiero hacer?

Un proyecto que genere instancias sinteticas que pueda vulnerar un sistema de autenticación biometrica.

## ¿Que necesito?

- El sistema de autentiación biometrica.
- Crear un lanzador de experimentos.
- Una GAN que genere instancias sinteticas.
- Métricas.

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

1. Seleccionar las caracteristicas más robustas con un algoritmo
1. Analizar la extracción de caracteristicas multiespectrales.

## Investigar algoritmos

**GA-LDA**:Se utiliza principalmente en clasificación y reducción de dimensionalidad. El objetivo de LDA es maximizar la varianza entre clases mientras minimiza la varianza dentro de cada clase.
