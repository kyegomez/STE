import torch 
import torch.nn as nn


class STEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(torch.clamp(input, min=-1.0, max=1.0))
    
    @staticmethod
    def backward(ctx, grad_output):
        #Bypass the non differterable operations
        return grad_output
    
class STE(nn.Module):
    def forward(self, input):
        return STEFunc.apply(input)
    

