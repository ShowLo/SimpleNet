# -*- coding: UTF-8 -*-

'''
测试分类
'''

from PIL import Image
from torchvision import transforms
import torch
from torch.autograd import Variable
import os
import json
from SimpleNet import SimpleNet

def predict_image(model, image_path):
    image = Image.open(image_path)

    # 测试时截取中间的90x90
    transformation1 = transforms.Compose([
        transforms.CenterCrop(90),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])

    ])

    # 预处理图像 
    image_tensor = transformation1(image).float()

    # 额外添加一个批次维度，因为PyTorch将所有的图像当做批次
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # 将输入变为变量
    input = Variable(image_tensor)

    # 预测图像的类别
    output = model(input)

    index = output.data.numpy().argmax()

    return index

if __name__ == '__main__':

    best_model_path = './output/epoch_462.pth'
    model = SimpleNet()
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    img_path = './images/test/bubble/066.jpg'
    predict_class = class_names[predict_image(model, img_path)]
    print(predict_class)