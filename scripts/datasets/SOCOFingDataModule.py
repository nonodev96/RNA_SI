import pytorch_lightning as pl
import torchvision.transforms as transforms


class SOCOFingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # The values 0.13070.1307 and 0.30810.3081 used in the normalization process for the MNIST dataset are significant because they represent the mean and standard deviation of the pixel values across the dataset.
                # Normalized value = Pixel value – Mean / Standard Deviation
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10
