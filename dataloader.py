# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches
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
        return self.n_samples
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]

dataset = WineDataset('./data/wine.csv')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

dataiter = iter(dataloader)
data = dataiter.next()

features, labels = data
print(features, labels)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4) # 4 is the batch size

print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):

        # ideally you now want to do = forward pass + backward pass + update weights

        # but this is just dummy data for the sake of explaining data loaders
        if (i+1) % 5 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)

# look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
inputs, targets = data
print(inputs.shape, targets.shape)