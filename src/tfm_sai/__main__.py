"""Entry point for tfm_sai."""

import argparse  # pragma: no cover

from tfm_sai.cli import main  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Input image dir. Default: input_dir",
        default="input_dir",
        required=True,
    )
    parser.add_argument(
        "--channels",
        help="Number of image channel",
        default=4,
    )
    parser.add_argument(
        "--not_cuda",
        action="store_true",
        help="disables cuda",
        default=0,
    )
    parser.add_argument(
        "--device",
        help="Device: cuda or cpu",
        default="cuda",
        choices=["cuda", "cpu"],
        required=True,
    )
    parser.add_argument("--epoch", default=5000)
    opt = parser.parse_args()
    main(opt)
