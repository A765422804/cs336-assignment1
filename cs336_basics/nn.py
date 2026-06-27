import torch
from torch import nn, Tensor
import math
from einops import einsum, rearrange
from jaxtyping import Float, Int

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        初始化权重W
        '''
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3*std,
            b=3*std
        )

    def forward(self, x:Float[Tensor, '... d_in'])->Float[Tensor, '... d_out']:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        初始化embedding表
        '''
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3,
            b=3
        )

    def forward(self, token_ids: Int[Tensor, '... d_in'])->Float[Tensor, '... d_in embedding_dim']:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        '''
        初始化RMSNorm的参数 g_i
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x:Float[Tensor, '... d_model'])->Float[Tensor, '... d_model']:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2,dim=-1, keepdim=True) + self.eps)

        result = (x / rms) * self.weight

        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        '''
        初始化创建三个线性逻辑
        '''
        super().__init__()
        
        if d_ff is None:
            d_ff = 64 * math.ceil( 8/3 * d_model / 64)

        self.linear_gate = Linear(d_model, d_ff, device, dtype)
        self.linear_down = Linear(d_ff, d_model, device, dtype)
        self.linear_up = Linear(d_model, d_ff, device, dtype)

    def forward(self, x:Float[Tensor, '... d_model'])->Float[Tensor, '... d_model']:
        gate = self.linear_gate(x)
        up = self.linear_up(x)
        hidden = gate * torch.sigmoid(gate) * up
        return self.linear_down(hidden)
        
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        构建RoPE中的cos和sin的缓存：
        1. 构建位置索引和频率索引
        2. 构建角度
        3. 构建cos和sin
        4. 存储为buffer
        '''
        super().__init__()

        assert(d_k % 2 == 0)

        # 创建位置索引
        positions = torch.arange(0, max_seq_len, device=device)

        # 创建频率索引
        dimension_indices = torch.arange(0, d_k // 2, device=device) 
        freqs = 1.0 / theta ** (2 * dimension_indices / d_k)

        # 构建角度
        angles = rearrange(positions, 'seq -> seq 1') * rearrange(freqs, 'dim -> 1 dim')

        # 构建cos和sin
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self, x: Float[Tensor, '... seq_len d_k'], token_positions: Int[Tensor, '... seq_len'])->Float[Tensor, '... seq_len d_k']:
        '''
        对当前输入的token进行旋转位置编码：
        1. 基于token_positions截取当前的token位置
        2. 拆分x成奇偶两部分
        3. 分别应用旋转公式
        4. 交替合并得到结果
        '''

        # 截取当前token位置
        cos = self.cos_cache[token_positions] # ..., seq_len, d_k // 2
        sin = self.sin_cache[token_positions] # ..., seq_len, d_k // 2

        # 拆分x成奇偶两部分
        x_even = x[..., 0::2] # ... seq_len d_k//2
        x_odd = x[..., 1::2] # ... seq_len d_k//2

        # 分别引用旋转公式
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        # 合并结果
        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd

        return out