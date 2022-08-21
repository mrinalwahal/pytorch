from operator import mod
import torch

""" x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

#   Generally, the last value will be a scalar value.
#   So that you can call backward pass.
y = y.mean()

y.backward()
print(x.grad)

#   To disable gradient function, use requires_grad=False
#   x.requires_grad_(False) """

weights = torch.ones(4, requires_grad=True)
#bias = torch.zeros(1, requires_grad=True)

for epoch in range(3):
    model_output = (weights + 3).sum()

    model_output.backward()

    print(weights.grad)

    #   To prevent the gradients from being accumulated over iterations, use zero_grad()
    #   Try running the code by commenting out the following line.
    weights.grad.data.zero_()