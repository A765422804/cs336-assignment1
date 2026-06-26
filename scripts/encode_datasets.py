'''
编码原始txt数据集到 token ids .npy
'''

from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
import numpy as np
from typing import Iterable

def encode_file(input_path: str, output_path: str, tokenizer: Tokenizer):
    '''
    调用两次tokenizer，第一次统计所有的token数量，然后创建等量大小的npy，第二次顺序写入npy
    '''
    chunk_size = 4 * 1024 * 1024
    special_token = '<|endoftext|>'
    num_tokens = 0

    # 第一次遍历文件，统计数量
    print(' - start first pass')
    token_iter = tokenizer.encode_iterable(iter_document_pieces(input_path, special_token, chunk_size))
    for _ in token_iter:
        num_tokens += 1
    print(' - num_tokens = ', num_tokens)

    # 根据num_tokens创建npy
    arr = np.lib.format.open_memmap(
        output_path,
        mode='w+',
        dtype=np.uint16,
        shape=(num_tokens,)
    )

    # 第二次遍历文件，写入npy
    print(' - start second pass')
    token_iter = tokenizer.encode_iterable(iter_document_pieces(input_path, special_token, chunk_size))
    for i, token_id in enumerate(token_iter):
        arr[i] = token_id

    arr.flush()
    assert(i + 1 == num_tokens)


def iter_document_pieces(input_path: str, special_token: str, chunk_size: int)->Iterable[str]:
    '''
    流式遍历整个txt文件，按照special_tokens迭代返回str
    '''
    buffer = ''
    with open(input_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)

            if chunk == '':
                if buffer != '':
                    yield buffer
                break

            buffer += chunk

            parts = buffer.split(special_token)
            for part in parts[:-1]:
                yield part + special_token

            buffer = parts[-1]

def main():

    # 创建目录
    Path('artifacts/tokenized').mkdir(parents=True, exist_ok=True)

    # 初始化tokenizer
    tinystories_tokenizer = Tokenizer.from_files('artifacts/tiny_stories_bpe_vocab.pkl', 'artifacts/tiny_stories_bpe_merges.pkl', ["<|endoftext|>"])
    owt_tokenizer = Tokenizer.from_files('artifacts/owt_bpe_vocab.pkl', 'artifacts/owt_bpe_merges.pkl', ["<|endoftext|>"])    

    # encode tinystories
    print('start encode tiny-train')
    encode_file('data/TinyStoriesV2-GPT4-train.txt', 'artifacts/tokenized/tinystories_train.npy', tinystories_tokenizer)

    print('start encode tiny-valid')    
    encode_file('data/TinyStoriesV2-GPT4-valid.txt', 'artifacts/tokenized/tinystories_valid.npy', tinystories_tokenizer)
    
    # encode owt
    print('start encode owt-train')
    encode_file('data/owt_train.txt', 'artifacts/tokenized/owt_train.npy', owt_tokenizer)

    print('start encode owt-valid')
    encode_file('data/owt_valid.txt', 'artifacts/tokenized/owt_valid.npy', owt_tokenizer)

if __name__ == '__main__':
    main()
