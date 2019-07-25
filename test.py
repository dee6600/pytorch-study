from __future__ import print_function
import torch

x = torch.rand(5,3, dtype= torch.double)
print(x)
print(x.size())

y = torch.rand(5, 3, dtype=torch.double)
print(y,y.size())

print(x + y)

print(y.add_(x))