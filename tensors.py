import torch
import numpy as np

#   Empty tensors
#   x = torch.empty(5, 3)

#   Random tensor
#   x = torch.rand(5, 3)

#  Tensors of Ones
#  x = torch.ones(5, 3)

#  Custom Tensor
x = torch.tensor([[1, 2], [3, 4]])

print(x)
#   print(x.dtype)
#   print(x.size())

y = torch.rand(2, 2)

#   Add Tensors
z = x + y

#  Print all columns for only row 0
print(x[0, :])

#   Reshape to one 1 dimensional tensor
x = x.view(4)

print(x)

#   Convert torch tensor numpy array
print(x.numpy())

#   Detect and run on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = y.to(device)
    z = x + y
    print(z)
