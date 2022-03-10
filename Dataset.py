from torch.utils.data import Dataset
import torch
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets

#数据集根地址
path = "dataset/dataset-resized"
#变换增强数据
transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomRotation((30,30)),
    transforms.RandomVerticalFlip(0.1),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
def getDataset():
    #分割数据集
    global path,transforms
    dataset = datasets.ImageFolder(path, transform=transforms)

    trainDataset,testDataset = torch.utils.data.random_split(dataset, [2000, 307])
    return trainDataset,testDataset


