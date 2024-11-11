
from tabulate import tabulate
from src.base.config import logger
from src.implementations.DCGAN import Discriminator


def info():
    dc = Discriminator()
    print(dc)


def prueba():
    import tensorflow as tf
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.build()
    model.summary()


def print_gpu_info():
    import nvsmi
    print("Nvidia SMI")
    print("-----------")
    print("GPU     count: ", nvsmi.get_gpus())
    print("GPU      info: ", list(nvsmi.get_available_gpus()))
    print("GPU processes: ", nvsmi.get_gpu_processes())


def print_debug():
    import torch
    print(torch.cuda.memory_summary())
    info_cuda = [
        ["torch.__version__", torch.__version__],
        ["torch cuda         is_available", torch.cuda.is_available()],
        ["torch cuda       current_device", torch.cuda.current_device()],
        ["torch cuda         device_count", torch.cuda.device_count()],
        ["torch cuda      get_device_name", torch.cuda.get_device_name(0)],
        ["torch cuda       is_initialized", torch.cuda.is_initialized()],
        ["torch cuda     memory_allocated", torch.cuda.memory_allocated()],
        ["torch cuda      memory_reserved", torch.cuda.memory_reserved()],
        ["torch cuda max_memory_allocated", torch.cuda.max_memory_allocated()],
        ["torch cuda  max_memory_reserved", torch.cuda.max_memory_reserved()]
    ]
    print(tabulate(info_cuda, headers=["Variable", "Value"]))

    logger.info('its working')


def main():
    # print_gpu_info()
    # print_debug()
    # prueba()
    info()


if __name__ == "__main__":
    main()
