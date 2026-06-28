import torch
from torch import nn, Tensor
import math
from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool

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
    
def silu(x):
    return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        '''
        初始化创建三个线性逻辑
        '''
        super().__init__()
        
        if d_ff is None:
            d_ff = 64 * math.ceil( 8/3 * d_model / 64)

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x:Float[Tensor, '... d_model'])->Float[Tensor, '... d_model']:
        gate = self.w1(x)
        up = self.w3(x)
        hidden = silu(gate) * up
        return self.w2(hidden)
        
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

def softmax(x: torch.Tensor, dim: int)->torch.Tensor:
    '''
    计算softmax，先减去dim的最大值，然后算exp，最后算dim的和
    '''
    x_shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(q: Float[Tensor, 'batch_size ... seq_len d_k'], k:Float[Tensor, 'batch_size ... seq_len d_k'], v:Float[Tensor, 'batch_size ... seq_len d_v'], mask:Bool[Tensor, 'seq_len seq_len'] | None = None)->Float[Tensor, 'batch_size ... seq_len d_v']:
    '''
    求注意力机制：
    1. QK^T
    2. / sqrt(d_k)
    3. mask处加负无穷表示遮蔽 (opt)
    4. softmax(dim=-1)
    5. @ V
    '''

    scores = einsum(q, k, '... q d_k, ... k d_k -> ... q k')
    scores /= math.sqrt(q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(~mask, -float('inf'))

    atten_probs = softmax(scores, -1)

    return einsum(atten_probs, v, '... q k, ... k d_v -> ... q d_v')

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int |None = None):
        '''
        构造多头自注意力线性层，包括Wq Wk Wv Wo
        因为规定 d_k = d_v = d_model / h 所以 h * dk = h * dv = d_model
        初始化RoPE
        '''
        super().__init__()

        assert(d_model % num_heads == 0)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.head_dim, max_seq_len)
        else:
            self.rope = None

    def forward(self, x: Float[Tensor, '... seq_len d_model'], token_positions: Int[Tensor, '... seq_len'] | None = None)->Float[Tensor, '... seq_len d_model']:
        '''
        多头自注意力的计算：
        1. 计算QKV
        2. 构建mask，和输入的device一致
        3. 拆分多头
        4. 对 qk 使用旋转位置编码
        5. 调用scaled_dot_product_attention
        6. 合并多头
        7. 使用Wo投影
        '''

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 拆分多头
        Q = rearrange(Q, '... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        K = rearrange(K, '... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim', num_heads=self.num_heads, head_dim=self.head_dim)
        V = rearrange(V, '... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim', num_heads=self.num_heads, head_dim=self.head_dim)

        # 旋转位置编码
        seq_len = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            else:
                token_positions = token_positions.unsqueeze(-2)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        mask = torch.tril(ones)

        out = scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        out = rearrange(out, '... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)', num_heads=self.num_heads, head_dim=self.head_dim)        

        return self.output_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len:int |None = None):
        '''
        transformer块的构造函数，需要构建一个attention layer和一个ffn layer，都采用pre-norm
        att layer:
            RMSNorm, MultiHeadSelfAttention
        ffn layer:
            RMSNorm, SwiGLU
        '''
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)

        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, '... seq_len d_model'], token_positions: Int[Tensor, '... seq_len'] | None = None)->Float[Tensor, '... seq_len d_model']:
        '''
        y = x + attn(ln1(x))
        z = y + ffn(ln2(y))
        '''

        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y)) 
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float):
        '''
        构建完整transformer模型，包括embedding， 多层transformer，一个RMS norm和一个线性层
        '''
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Int[Tensor, 'batch_size seq_len'], token_positions: Int[Tensor, '... seq_len'] | None = None)->Float[Tensor, 'batch_size seq_len vocab_size']:
        '''
        embedding, transformer_blocks, RMSNorm, lm_head
        '''
        x = self.token_embeddings(token_ids)
        for block in self.layers:
            x = block(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

        
