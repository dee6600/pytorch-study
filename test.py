from __future__ import print_function
import torch

x = torch.tensor([5.5,3])
print(x)
print(x.size())

x = x.new_ones(5, 3, dtype=torch.double)
print(x,x.size())