from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('~/PycharmProjects/TFM_SAI/scripts/eq15.png')
model = LatexOCR()
print(model(img))