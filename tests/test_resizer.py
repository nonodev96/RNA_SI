import os
import cv2
from tfm_sai.resizer import box

CURRENT_WORKING_DIRECTORY = os.getcwd()


def test_box():
    x = 0.7
    box_rate = box(x)
    assert box_rate == 0.0
