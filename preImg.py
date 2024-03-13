# 保存训练好的模型
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import LoadData

import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from torchvision.models import alexnet  #最简单的模型
from torchvision.models import vgg11, vgg13, vgg16, vgg19   # VGG系列
from torchvision.models import resnet18, resnet34,resnet50, resnet101, resnet152    # ResNet系列
from torchvision.models import inception_v3     # Inception 系列
# # 读取训练好的模型，加载训练好的参数
# model = resnet18()
# model.load_state_dict(torch.load("model_resnet101.pth"))
import cv2
# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# # 读取训练好的模型，加载训练好的参数
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))  


# # example_input = torch.rand(1, 3, 320, 320).cuda()
# example_input = torch.rand(1, 3, 320, 320)
# output = model(example_input)
# print(output) l  L

model = torch.load('model.pth')  # 直接加载模型
model.eval()

# example_input = torch.rand(1, 3, 320, 320).cuda()
# output = model(example_input)
# print(output)
def preImg1(image_cv2,num):
    # image_path = "12.jpg"
    # img = Image.open(image_path).convert('RGB')
    # img = image_path.convert('RGB')
    # 获取图像的宽度和高度
    # 将图像从BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # 创建Pillow图像对象
    img = Image.fromarray(image_rgb)

    width, height = img.size

    if (height == 720):
        image1 = img.crop((345, 155, 825, 700))
    elif (height == 1440):
        image1 = img.crop((690, 310, 1650, 1090))
    new_size = (160, 160)
    # 调整图像大小
    image = image1.resize(new_size)


    # 图片标准化
    transform_BZ= transforms.Normalize(
    mean= [0.09978075, 0.09978075, 0.09978075],
    std= [0.31588092, 0.31588092, 0.31588092]
    )

    train_tf = transforms.Compose([
                    # transforms.Resize(224),#将图片压缩成224*224的大小
                    # transforms.RandomHorizontalFlip(),#对图片进行随机的水平翻转
                    # transforms.RandomVerticalFlip(),#随机的垂直翻转
                    transforms.ToTensor(),#把图片改为Tensor格式
                    transform_BZ#图片标准化的步骤
                ])

    img1 = train_tf(image)
    # img_tensor = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False).cuda()


    img_normalize = torch.unsqueeze(img1, 0).cuda()

    output = model(img_normalize)
    print(output)

    output1 = torch.softmax(output,dim=1)
    print(output1)

    pred_value, pred_index = torch.max(output1, 1)
    print(pred_value)
    print(pred_index)

    pred_value = pred_value.detach().cpu().numpy()
    pred_index = pred_index.detach().cpu().numpy()

    num = pred_index[0]

    print(pred_value)
    print(pred_index)

    classes = ["无鱼", "有鱼"]

    print("预测类别为： ", classes[pred_index[0]], " 可能性为: ", pred_value[0] * 100, "%")



