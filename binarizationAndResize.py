# 图片二值化的代码，0 1的地方可能要换一下
from PIL import Image
import os

path = "data"  # 图像读取地址
for root, dirs, files in os.walk(path):
    for f in files:
        jpg_name = os.path.join(root, f)
        print(jpg_name)
        img = Image.open(jpg_name)
        Img = img.convert('L')
        threshold = 200
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)

        photo = Img.point(table, '1')
        photo.resize((100, 100), Image.ANTIALIAS)
        photo.save(jpg_name)
