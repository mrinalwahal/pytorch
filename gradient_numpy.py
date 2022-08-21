#   This example shows usage of gradient descent in Numpy
#   to multiple input numbers by 2.

import numpy as np

#  f = w + x

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

#   model prediction
def forward(x):
    return x * w

#   loss = Mean Squared Error (MSE)
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

#   gradient of loss with respect to w
#   MSE = 1/N * (w*x - y)**2
#   dJ/dw = 1/N * 2 * (w*x - y) * x
def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()

print(f'Prediction before training: {forward(5):.3f}')

# Training parameters
learning_rate = 0.01
n_iters = 20

#   Training loop
for epoch in range(n_iters):

    y_predicted = forward(X)
    l = loss(Y, y_predicted)

    #   gradients
    dw = gradient(X, Y, y_predicted)

    #   update weights
    #   negative sign is for opposite direction of gradient descent
    w = w - learning_rate * dw

    #   Only print every second step
    if epoch % 2 == 0:
        print(f'{epoch+1:02d}: w = {w:.2f}, loss = {l:.6f}')

print(f'Prediction after training: {forward(5):.3f}')

