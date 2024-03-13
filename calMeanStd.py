import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

def calculate_mean_std(image_list):
    pixel_values = []
    max_width = 0
    max_height = 0

    for image_path in image_list:
        img = Image.open(image_path).convert('RGB')

        # 获取图像的宽度和高度
        width, height = img.size
        print(image_path)
        if (height == 720):
            image1 = img.crop((345, 155, 825, 700))
        elif (height == 1440):
            image1 = img.crop((690, 310, 1650, 1090))
        new_size = (320, 320)
        # 调整图像大小
        image = image1.resize(new_size)
        # 调整图像大小
        # image = img2.resize((img2.size[0] // 2, img2.size[1] // 2), Image.ANTIALIAS)
        # width, height = image.size
        max_width = 320
        max_height = 320
        pixel_values.append(image)

    pixel_array = np.zeros((len(pixel_values), max_height, max_width, 3), dtype=np.float32)

    for i, image in enumerate(pixel_values):
        image = image.resize((max_width, max_height), Image.BILINEAR)
        image_array = np.array(image, dtype=np.float32)

        pixel_array[i] = image_array
    # 归一化图像数据
    pixel_array /= 255.0
    mean = np.mean(pixel_array, axis=(0, 1, 2))
    std = np.std(pixel_array, axis=(0, 1, 2))

    return mean, std


# 从txt文件中读取图像地址列表
def read_image_list_from_txt(file_path):
    image_list = []
    b = 0
    with open(file_path, 'r') as f:
        for line in f:
            b = b+1
            image_path1 = line.strip()
            image_path = image_path1.split()
            a = image_path[0]
            if os.path.isfile(a):
                image_list.append(a)
            else:
                print(f"Image file not found: {image_path}")

    return image_list


# 指定包含图像地址的txt文件路径
txt_file_path = 'train.txt'

# 从txt文件中读取图像地址列表
image_list = read_image_list_from_txt(txt_file_path)

# 计算数据集的均值和标准差
mean, std = calculate_mean_std(image_list)

print("Mean:", mean)
print("Std:", std)