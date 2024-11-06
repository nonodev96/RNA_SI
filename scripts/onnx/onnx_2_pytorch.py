import torch
import onnx
import onnxruntime as ort
from onnx2pytorch import ConvertModel


# Cargar el modelo ONNX
model_path = './benign-dcgan-mnist.onnx'
onnx_model = onnx.load(model_path)

# Convert ONNX model to PyTorch
pytorch_model = ConvertModel(onnx_model)

torch.save(pytorch_model.state_dict(), "benign-dcgan-mnist.pth")
