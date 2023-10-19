import torch 
from ste.main import STE

# random input
x = torch.randn(1, 3, 32, 32)

# STE
ste = STE()

# forward
y = ste(x)

print(y)