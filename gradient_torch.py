#   This example shows usage of gradient descent in Pytorch
#   to multiple input numbers by 2.

import torch

#  f = w + x

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#   model prediction
def forward(x):
    return x * w

#   loss = Mean Squared Error (MSE)
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

print(f'Prediction before training: {forward(5):.3f}')

# Training parameters
learning_rate = 0.01
n_iters = 100

#   Training loop
for epoch in range(n_iters):

    y_predicted = forward(X)
    l = loss(Y, y_predicted)

    #   calculate gradients
    l.backward()

    #   update weights
    with torch.no_grad():
    
        #   negative sign is for opposite direction of gradient descent
        w = w - learning_rate * w.grad

    #   zero gradients
    w.grad.zero_()

    #   Only print every 10th step
    if epoch % 10 == 0:
        print(f'{epoch+1:02d}: w = {w:.2f}, loss = {l:.6f}')

print(f'Prediction after training: {forward(5):.3f}')

