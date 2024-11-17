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

## Lista tareas

- [DONE] Solicitar bases de datos CASIA Y GPSD.
- [DONE] Crear la carga de datos.
- [DONE] Añadir tipos de GANs
  - [DONE] BEGAN
  - [DROP] CGAN
  - [DONE] DCGAN
  - [DONE] GAN
  - [DONE] WGAN
  - [DONE] WGAN_GP

- [DONE] Test ART DGM Devil In GAN
- [DONE] Implementar **ReD : REtraining with Distillation** en pytorch siguiendo la arquitectura del proyecto ART
  - [DROP][No funciono] Pasar la GAN MNIST benigna de tensorflow a onnx 
  - [DROP][No funciono] Pasar la GAN MNIST benigna de onnx a pytorch
  - [DONE] Comprobar que la GAN en pytorch sigue funcionando igualmente.
  - [DONE] Crear el lanzador de experimentos.

- [DONE] Implementar el estimador/generation de pytorch
- [DONE] Implementar **TrAIL: Training with AdversarIal Loss** en pytorch siguiendo la arquitectura del proyecto ART
  - [DONE] Comprobar que la GAN en pytorch sigue funcionando igualmente.


## Notas

```python
# Define the trigger
z_trigger = np.random.randn(1, 100).astype(np.float64)

# Set the target the trigger
x_target = np.random.randint(low=0, high=256, size=(28, 28, 1)).astype("float64")
x_target = (x_target - 127.5) / 127.5
x_target_tf = tf.cast(np.arctanh(0.999 * x_target), tf.float64)
```