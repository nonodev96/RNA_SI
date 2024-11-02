from PIL import Image
from pix2tex.cli import LatexOCR

img = Image.open('~/PycharmProjects/RNA_SI/scripts/eq15.png')
model = LatexOCR()
print(model(img))