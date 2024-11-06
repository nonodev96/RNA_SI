import tensorflow as tf
import onnx
import tf2onnx

# Convert the model to ONNX format
loaded_model = tf.keras.models.load_model('../devil-in-gan/data/art-dgm-ipynb-data/benign-dcgan-mnist')
loaded_model.summary()

onnx_model, _ = tf2onnx.convert.from_keras(loaded_model)

model_path = './benign-dcgan-mnist.onnx'
onnx.save(onnx_model, model_path)