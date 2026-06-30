from cs336_basics.nn import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer
import torch

def generate_text(model: TransformerLM, tokenizer : Tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p :float, eos_token:str="<|endoftext|>"):
    """
    1. 把用户输入prompt和eos_token转为token ids
    2. 开始迭代生成：
        1. 把当前最大长度的内容输入到模型，得到logits
        2. 取logits最后一项，应用temperature scaling和Top-p sampling
        3. 得到token id后把它再放到当前输入token的末尾
        4. 循环以上三步，直到输出的是eos或者达到最大输出次数
    3. 将生成的token ids解码成字符输出
    """

    # encode
    prompt_token_ids = tokenizer.encode(prompt)
    eos_token_id = tokenizer.encode(eos_token)[0] # 单个元素list，直接取值

    # 必要参数
    device = next(model.parameters()).device
    context_length = model.context_length

    # 迭代生成
    generate_token_list = []
    all_token_list = prompt_token_ids.copy()
    model.eval()
    while len(generate_token_list) < max_new_tokens:
        with torch.no_grad():
            input_tokens = torch.tensor(all_token_list[-context_length:], device=device, dtype=torch.long)
            input_tokens = input_tokens.unsqueeze(0)
            logits = model(input_tokens) # 1 seq_len vocab_size

        # 取最后一项
        last_logtis = logits[0, -1, :] # vocab_size

        # 应用temperature scaling的softmax
        scaled_logits = last_logtis / temperature
        probs = softmax(scaled_logits, -1)

        # 应用top-p截断
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p 
        mask[1:] = mask[: -1]
        mask[0] = False
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / torch.sum(sorted_probs)

        # 采样
        sampled_position = torch.multinomial(sorted_probs, num_samples=1)
        next_token_id = sorted_indices[sampled_position]
        next_id = next_token_id.item()

        generate_token_list.append(next_id)
        if next_id == eos_token_id:
            break

        all_token_list.append(next_id)

    return tokenizer.decode(generate_token_list)