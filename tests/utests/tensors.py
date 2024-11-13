import numpy as np
import torch


def test_tensor_tensorflow():
    # TENSORFLOW
    import tensorflow as tf

    a = tf.constant([-5, -7, 2, 5, 7], dtype=tf.float64)
    b = tf.constant([1, 3, 9, 4, 7], dtype=tf.float64)
    print("Tensorflow tensor a: ", a)
    print("Tensorflow tensor b: ", b)

    res_squared_difference = tf.math.squared_difference(a, b)
    res_mean = tf.math.reduce_mean(res_squared_difference)
    print("Tensorflow tensor Result squared_difference: ", res_squared_difference)
    print("Tensorflow tensor Result mean: ", res_mean)


def test_tensor_pytorch():
    # PYTORCH

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

    torch_res_squared_difference_tensor = torch_res_squared_difference.to(torch.float64)
    torch_res_mean = torch.mean(torch_res_squared_difference_tensor)
    print("Torch tensor Result squared_difference: ", torch_res_squared_difference)
    print("Torch tensor Result mean: ", torch_res_mean)


def main() -> None:
    test_tensor_tensorflow()
    test_tensor_pytorch()


if __name__ == "__main__":
    main()
