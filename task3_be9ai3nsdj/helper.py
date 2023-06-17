import numpy as np
import torch
from torchvision import transforms
import torchvision.datasets as datasets

def calculate_mean_and_std():
    mean = np.zeros(3)
    std = np.zeros(3)
    cnt = 0

    # 遍历数据集中的所有图像
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),])
    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    for img, idx in train_dataset:
        cnt += 1
        if(cnt % 500 == 0):
            print(cnt)
        for i in range(3):
            mean[i] += torch.mean(img[i,:,:])
            std[i] += torch.std(img[i,:,:])

    # 计算均值和标准差的平均值
    mean /= len(train_dataset)
    std /= len(train_dataset)

    return mean, std

mean, std = calculate_mean_and_std()
print(mean)
print(std)