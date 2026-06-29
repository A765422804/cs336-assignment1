import numpy as np
from torch import Tensor
from jaxtyping import Int
import torch

def get_batch(x : np.ndarray, batch_size: int, context_length: int, device: str)->tuple[Int[Tensor, 'batch_size seq_len'], Int[Tensor, 'batch_size seq_len']]:
    '''
    x是输入的数据集，包括原始文本的所有tokens id序列，现在需要随机采样batch_size个序列的输入输出对，每个长度为context len
    '''
    text_len = len(x)
    starts = np.random.randint(low=0, high=text_len - context_length, size=batch_size) # B,
    offsets = np.arange(0, context_length) # T,
    indices = starts[:, None] + offsets[None,:] # B,T + B, T
    inputs_np = x[indices]
    targets_np = x[indices + 1]

    return torch.from_numpy(inputs_np).long().to(device), torch.from_numpy(targets_np).long().to(device)