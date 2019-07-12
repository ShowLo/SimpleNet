# -*- coding: UTF-8 -*-

'''
负责训练及测试数据的读取
'''

from torchvision import transforms, datasets
import os
import torch
from PIL import Image

def readImg(path):
    '''
    用于替代ImageFolder的默认读取图片函数，以读取单通道图片
    '''
    return Image.open(path)

def ImageDataset(args):
    # 数据增强及归一化
    # 图片都是100x100的，训练时随机裁取90x90的部分，测试时裁取中间的90x90
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(90),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    }

    data_dir = args.data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x], loader=readImg)
                    for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                shuffle=(x == 'train'), num_workers=args.num_workers)
                for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names