# TODO List

## ¿Que quiero hacer?

Un proyecto que genere instancias alteradas que pueda vulnerar un sistema de autenticación biométrica.

Necesito hacer un ataque de envenenamiento de una GAN.

Segun un disparador se active y cree una instancia alterada por un envenenamiento en un re entrenamiento.

## ¿Que necesito?

- UN lanzador de experimentos.
- Cargar modelos Devil In GAN (IBM)
- Métricas.

## Conjuntos de datos

- MNIST
- SOCOFing
- CASIA PalmprintV1
- CASIA Multi-Spectral PalmprintV1
- Hands GPDS 100 Contactless 2 bands

## Modelos

- MNIST Benigna

## Lista tareas

- [DONE] Solicitar bases de datos CASIA Y GPSD.
- [DONE] Crear la carga de datos.
- [DONE] Crear el lanzador de experimentos.
- [TODO] Test ART DGM Devil In GAN
- [TODO] Implementar **ReD : REtraining with Distillation** en pytorch siguiendo la arquitectura del proyecto ART
  - [TODO] Pasar la GAN MNIST benigna de tensorflow a onnx
  - [TODO] Pasar la GAN MNIST benigna de onnx a pytorch
  - [TODO] Comprobar que la GAN en pytorch sigue funcionando igualmente.
- [TODO] Implementar el estimador/generation de pytorch
