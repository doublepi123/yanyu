import numpy as np
from PIL import Image
import os

path = "photo/"
for root, dirs, files in os.walk(path):
    for f in files:
        img = Image.open(os.path.join(root, f))
        img = img.resize((100, 100), Image.ANTIALIAS)
        img = np.asarray(img)
        print(img)
