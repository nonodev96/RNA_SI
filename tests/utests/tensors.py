import numpy as np
import torch


def test_tensorflow_tensor():
    # TENSORFLOW
    import tensorflow as tf

    a = tf.cast(np.array([-5, -7, 2, 5, 7]), dtype=tf.float64)
    b = tf.cast(np.array([1, 3, 9, 4, 7]), dtype=tf.float64)
    print("Tensorflow tensor a: ", a)
    print("Tensorflow tensor b: ", b)

    res_squared_difference = tf.math.squared_difference(a, b)
    res_mean = tf.math.reduce_mean(res_squared_difference)
    print("Tensorflow tensor Result squared_difference: ", res_squared_difference)
    print("Tensorflow tensor Result mean: ", res_mean)


def test_pytorch_tensor():
    # PYTORCH
    import torch

    # Option 1
    # torch_a = torch.tensor(np.array([-5, -7, 2, 5, 7]), dtype=torch.float64)
    # torch_b = torch.tensor(np.array([1, 3, 9, 4, 7]), dtype=torch.float64)

    # Option 2
    # torch_a = torch.from_numpy(np.array([-5, -7, 2, 5, 7])).double()
    # torch_b = torch.from_numpy(np.array([1, 3, 9, 4, 7])).double()

    # Option 3
    torch_a = torch.tensor(np.array([-5, -7, 2, 5, 7]), dtype=torch.float64)
    torch_b = torch.from_numpy(np.array([1, 3, 9, 4, 7])).double()
    print("Torch tensor a: ", torch_a)
    print("Torch tensor b: ", torch_b)
    torch_res_squared_difference = (torch_a - torch_b) ** 2
    torch_res_squared_difference_v2 = (torch_a - torch_b).pow(2)

    torch_res_squared_difference_tensor = torch_res_squared_difference.to(torch.float64)
    torch_res_mean = torch.mean(torch_res_squared_difference_tensor)
    print("Torch tensor Result squared_difference: ", torch_res_squared_difference)
    print("Torch tensor Result squared_difference: ", torch_res_squared_difference_v2)
    print("Torch tensor Result mean: ", torch_res_mean)


def test_pytorch_dataloader():
    # Ejemplo de datos (tensores en PyTorch)
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    x_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_data = torch.tensor([0, 1, 0])

    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(type(dataloader))

    for batch in dataloader:
        print(batch)

def test_pytorch_dataloader_mnist():
    # Ejemplo de datos (tensores en PyTorch)
    import torch
    import torchvision

    from torch.utils.data import TensorDataset, DataLoader
    dataset = torchvision.datasets.MNIST(
        root="../../datasets/mnist",
        download=True,
        transform=torchvision.transforms.Compose(
            transforms=[
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for i, (images_batch, _) in enumerate(dataloader):
        print(images_batch.shape)


def main() -> None:
    test_tensorflow_tensor()
    test_pytorch_tensor()
    # test_pytorch_dataloader_mnist()


if __name__ == "__main__":
    main()
