import numpy as np

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
            transforms.Resize((28, 28)),
        ]
    )

    img_tensor = transform(image)

    print(img_tensor.numpy())
    print(img_tensor.shape)
    # print(np.save("./data/x-target/shin-chan-1x28x28.npy", img_tensor.numpy()))


def main():
    debug_print()
    generate_x_target()


if __name__ == "__main__":
    main()
