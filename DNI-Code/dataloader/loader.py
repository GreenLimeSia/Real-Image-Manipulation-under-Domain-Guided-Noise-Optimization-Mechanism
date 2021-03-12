import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms


class CelebA_HQ(Dataset):
    def __init__(self, data_path, attr_path, selected_attrs):
        super(CelebA_HQ, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2, dtype=torch.float)
        return img, att

    def __len__(self):
        return self.length


def get_loader(image_dir, label_dir, batch_size):
    selected_attrs = ["Eyeglasses", "Male", "Mouth_Slightly_Open", "Smiling", "Young"]
    dataset = CelebA_HQ(image_dir, label_dir, selected_attrs)
    # split data ito train, valid, test set 7:2:1
    indices = list(range(30000))
    split_train = 21000
    split_valid = 27000
    train_idx, valid_idx, test_idx = indices[:split_train], indices[split_train:split_valid], indices[split_valid:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    validloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

    return trainloader, validloader, testloader
