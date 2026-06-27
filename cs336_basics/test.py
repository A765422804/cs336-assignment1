import torch
from einops import rearrange, einsum

D = torch.ones(size=(3,4,6))
A = torch.ones(size=(5, 6))

Y = D @ A.T

print(Y.shape)

Y = einsum(D, A, 'batch sequence d_in, d_out d_in -> batch sequence d_out')

print(Y.shape)