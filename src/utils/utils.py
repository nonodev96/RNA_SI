import torch
from skimage import io as skimage
from tabulate import tabulate


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def print_cuda_info():
    import torch
    print(torch.cuda.memory_summary())
    info_cuda = [
        ["torch.__version__", torch.__version__],
        ["torch cuda is_available", torch.cuda.is_available()],
        ["torch cuda current_device", torch.cuda.current_device()],
        ["torch cuda device_count", torch.cuda.device_count()],
        ["torch cuda get_device_name", torch.cuda.get_device_name(0)],
        ["torch cuda is_initialized", torch.cuda.is_initialized()],
        ["torch cuda memory_allocated", torch.cuda.memory_allocated()],
        ["torch cuda memory_reserved", torch.cuda.memory_reserved()],
        ["torch cuda max_memory_allocated", torch.cuda.max_memory_allocated()],
        ["torch cuda max_memory_reserved", torch.cuda.max_memory_reserved()],

        ["torch backends cpu    get_cpu_capability", torch.backends.cpu.get_cpu_capability()],
        ["torch backends cudnn  is_available", torch.backends.cudnn.is_available()],
        ["torch backends mkl    is_available", torch.backends.mkl.is_available()],
        ["torch backends mkldnn is_available", torch.backends.mkldnn.is_available()],
        ["torch backends mps    is_available", torch.backends.mps.is_available()],
        ["torch backends openmp is_available", torch.backends.openmp.is_available()],
    ]
    print(tabulate(info_cuda, headers=["Variable", "Value"]))


def print_model_summary(model: torch.nn, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "output_shape": list(output.size()),
                "nb_params": sum(p.numel() for p in module.parameters())
            }
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != model:
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    model.apply(register_hook)
    with torch.no_grad():
        model(torch.zeros(1, *input_size))

    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    for layer in summary:
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"])
        )
        total_params += summary[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print(f"Total params: {total_params:,}")
    print("----------------------------------------------------------------")


def imreadImg(path, opt):
    x = skimage.imread(path)
    x = x.astype(float)
    if x.ndim == 2:
        x = x[:, :, None, None]
        max_x = x.max()
        x = x.transpose(2, 3, 0, 1) / max_x
    elif x.ndim == 3:
        x = x[:, :, :, None]
        x = x.transpose(3, 2, 0, 1)
        max_x = x.max()
        x = x / max_x
    x = torch.from_numpy(x)
    x = x.to(opt.device).type(torch.cuda.FloatTensor)
    x = (x - 0.5) * 2
    x = x.clamp(-1, 1)
    return x, max_x


def truncate_string(s, max_length=50):
    if len(s) <= max_length:
        return s
    else:
        return s[:max_length] + "..."
