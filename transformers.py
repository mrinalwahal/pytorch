'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

class WineDataset(Dataset):

    #   Data loading
    def __init__(self, file, transform=None):

        xy = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float32)

        #   since this tutorial is on transformers,
        #   we are not going to convert this into a tensor,
        #   and leave it as a numpy array
        self.data = xy[:, 1:]
        self.target = xy[:, [0]]

        self.n_samples = xy.shape[0]

        self.transform = transform

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = self.data[index], self.target[index]
        
        #   apply transform, if available
        if self.transform:
            sample = self.transform(sample)

        return sample


#   Create custom transformer
class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MultiplicationTransform():
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor

        return inputs, targets

dataset = WineDataset('./data/wine.csv', transform=ToTensor())

first_data = dataset[0]
features, labels = first_data

print(type(features), type(labels))

