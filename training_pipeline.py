#   This example shows usage of gradient descent in Pytorch
#   to multiple input numbers by 2.

import torch
import torch.nn as nn

#  f = w + x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[3],[6],[9],[12]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

#   Print samples and features
print(f'Samples: {n_samples}')
print(f'Features: {n_features}')

input_size = n_features
output_size = n_features

#   model = nn.Linear(input_size, output_size)
#   You can also write a custom model class.
class LinearRegression(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: {model(X_test).item():.3f}')

# Training parameters
learning_rate = 0.01
n_iters = 1000

#   Define loss
loss = nn.MSELoss()

#   Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#   Training loop
for epoch in range(n_iters):

    y_predicted = model(X)
    l = loss(Y, y_predicted)

    #   calculate gradients
    l.backward()

    #   update weights
    optimizer.step()

    #   zero gradients
    optimizer.zero_grad()

    #   Only print every 10th step
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'{epoch+1:02d}: w = {w[0][0].item():.2f}, loss = {l:.6f}')

print(f'Prediction after training: {model(X_test).item():.3f}')

