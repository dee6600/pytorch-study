from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

y = torch.rand(5, 3)
print(y)

z = torch.zeros(5, 3, dtype=torch.long)
print(z)
