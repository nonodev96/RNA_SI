from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision

from src.utils.utils import print_cuda_info


def debug_print():
    print_cuda_info()


def generate_x_target():
    from PIL import Image
    import torchvision.transforms as transforms

    image = Image.open("./data/x-target-img/shin-chan.png")

    transform = transforms.Compose(
        transforms=[
            # Escala de grises con 1 canal
            # transforms.Grayscale(num_output_channels=3),
            transforms.PILToTensor(),
            transforms.Resize((64, 64)),
        ]
    )

    img_tensor = transform(image)

    # print(np.save("./data/x-target/shin-chan-3x64x64.npy", img_tensor.numpy()))
    # torchvision.utils.save_image(img_tensor.to("cpu", torch.uint8).numpy(), f"./results/images/x-target-torchvision.png")

    # Se espera que el tensor tenga un formato (height, width, channels), pero el tensor que estás pasando tiene el formato (channels, height, width), típico de PyTorch.
    plt.imshow(img_tensor.numpy().transpose(1, 2, 0))
    plt.savefig(f"./results/images/x-target-npz.png")


def main():
    # debug_print()
    generate_x_target()


if __name__ == "__main__":
    main()
