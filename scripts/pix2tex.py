from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('/home/amachuca/PycharmProjects/TFM_SAI/scripts/eq15.png')
model = LatexOCR()
print(model(img))