# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us


'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''

import torch
import torchvision
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

class WineDataset(Dataset):

    #   Data loading
    def __init__(self, file):
        xy = np.loadtxt(file, delimiter=',', skiprows=1, dtype=np.float32)

        #   split data
        self.data = torch.from_numpy(xy[:, 1:])
        self.target = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __len__(self):
        return len(self.n_samples)
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]

dataset = WineDataset('./data/wine.csv')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

dataiter = iter(dataloader)
data = dataiter.next()

features, labels = data
print(features, labels)