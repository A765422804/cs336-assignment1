import torch
import os
import typing

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, interation: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    '''
    存储模型中间状态，包括模型参数、优化器参数和迭代次数
    '''

    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    checkpoint = {
        'model': model_state,
        'optimizer': optimizer_state,
        'iteration': interation
    }

    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer)->int:
    '''
    加载模型中间状态，包括模型参数优化器参数，返回迭代次数
    '''

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['iteration']