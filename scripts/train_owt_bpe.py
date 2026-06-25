from cs336_basics import bpe
import pickle
from pathlib import Path

def main():
    '''
    1. 训练owt
    2. 保存结果
    3. 输出最长vocab
    '''
    input_path = 'data/owt_valid.txt'
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = bpe.train_bpe(input_path, vocab_size, special_tokens)
    print('len(vocab): ' , len(vocab))
    print('len(merges): ' , len(merges))

    Path('artifacts').mkdir(exist_ok=True)

    with open('artifacts/owt_bpe_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    with open('artifacts/owt_bpe_merges.pkl', 'wb') as f:
        pickle.dump(merges, f)  

    longest_token = b''
    for token_bytes in vocab.values():
        if len(token_bytes) > len(longest_token):
            longest_token = token_bytes

    print('longest token:')
    print(repr(longest_token))
    print(len(longest_token))
    print(longest_token.decode('utf-8', errors='replace'))

if __name__ == '__main__':
    main()