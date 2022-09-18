#   1. Design model (input, output, forward pass)
#   2. Define loss function and optimizer
#   3. Train model
#       - Forward pass: compute prediction and loss
#       - Backward pass: compute gradients
#       - Update weights
#   4. Plot predictions

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare datasets
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, random_state=4, noise=20)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

#   reshape as column vector
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape
input_size = n_features
output_size = 1

#   Model
class LinearRegression(nn.Module):
    
        def __init__(self, input_size, output_size):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(input_size, output_size)
    
        def forward(self, x):
            return self.linear(x)

model = LinearRegression(input_size, output_size)

#   Loss function
loss = nn.MSELoss()

#   Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#   Training loop
for epoch in range(200):
    
        y_predicted = model(X)
        l = loss(y_predicted, y)
    
        #   calculate gradients
        l.backward()
    
        #   update weights
        optimizer.step()
    
        #   zero gradients
        optimizer.zero_grad()
    
        #   Only print every 10th step
        if epoch % 10 == 0:
            print(f'{epoch+1:02d}: w = {model.linear.weight.item():.2f}, loss = {l.item():.6f}')

#   Plot predictions
predicted = model(X).detach().numpy()

""" plt.scatter(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show() """