from torch import optim
from typing import Optional
from collections.abc import Callable, Iterable
import math
import torch

class AdamW(optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 0.01):
        '''
        初始化AdamW，包括输入参数和若干超参数
        '''
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps' : eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        '''
        一次adamW的迭代：
        1. 遍历所有的param_groups，其中包括需要优化的参数['params']和所有的超参数
        2. 遍历所有的优化参数（没有梯度就跳过），获取当前优化参数的state里面的各种状态，如果没有先初始化
        3. 计算当前参数更新所需的变量
        4. 应用权重衰减和
        5. 更新一阶矩和二阶矩
        6. 应用梯度下降
        7. 把更新后的参数写回state
        
        '''
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                
                t = state.get('t', 1)
                alpha_t =  lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                p.data -= lr * weight_decay * p.data # 应用权重衰减

                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2

                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss

def cos_learning_rate_schedule(t:int, alpha_max: float, alpha_min: float, T_w: int, T_c:int)->float:
    '''
    包括warm up | cosine annealing | post-annealing三个阶段
    '''

    if t < T_w:
        return t / T_w * alpha_max
    elif t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w)/ (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min
    
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm:float, eps: float = 1e-6):
    '''
    输入所有参数的梯度，然后计算整体的l2_norm，如果超过了阈值，就应用梯度裁剪
    计算整体的l2 norm相当于把所有的参数的梯度展平乘一个大向量，然后每个元素平方求和再开根号，所以实际计算可以直接算平方
    '''
    params = list(params)

    l2_norm_square = 0
    for param in params:
        if param.grad is not None:
            l2_norm_square += torch.sum(param.grad.data ** 2)
    
    if l2_norm_square >= max_l2_norm ** 2:
        factor = max_l2_norm / (torch.sqrt(l2_norm_square) + eps)

        for param in params:
            if param.grad is not None:
                param.grad.data *= factor
