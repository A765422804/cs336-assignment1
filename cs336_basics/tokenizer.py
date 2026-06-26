import pickle
from typing import Iterable
import regex as re

# pre-tokenization 使用的正则匹配项
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        '''
        初始化vocab、merges和special_token，以及构建反向映射和merge-ranks，方便后面pretoken merge，以及排序后的special_tokens
        '''
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for i, merge in enumerate(merges):
            self.merge_ranks[merge] = i
        self.sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        # 构建反向映射
        self.bytes_to_id:dict[bytes, int] = {}
        for token_id, token_bytes in vocab.items():
            self.bytes_to_id[token_bytes] = token_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens:list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)  

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str)->list[int]:
        '''
        将字符串编码成token：
        1. 将字符串转化为pretoken
        2. 对于special_token直接转化，对于非special_token，调用工具函数去处理
        '''

        tokens:list[int] = []

        if self.special_tokens:
            pattern = '(' + '|'.join([re.escape(special_token) for special_token in self.sorted_special_tokens]) + ')'
            text_segments = re.split(pattern, text)
        else:
            text_segments = [text]

        for text_segment in text_segments:
            if text_segment == '':
                continue
            elif text_segment in self.special_tokens:
                tokens.append(self.bytes_to_id[text_segment.encode(encoding='utf-8')])
            else:
                tokens.extend(self._encode_text_segment(text_segment))

        return tokens

    def encode_iterable(self, iterable: Iterable[str])->Iterable[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids:list[int])->str:
        '''
        将token解码成字符串，首先需要将token映射到bytes，然后把bytes拼在一起成为一个单独的bytes，最后decode成utf-8，error = replace
        '''

        token_bytes = b''.join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode(encoding='utf-8', errors='replace')

    def _encode_text_segment(self, text_segment: str)->list[int]:
        '''
        输入是不含special_token的字符串，需要输出对应的token序列：
        1. 通过PAT做pretoken处理
        2. 对于每个，拆成tuple[bytes, ...]
        3. 每个pretoken内部按照merge顺序合并，具体来说，每次取当前相邻pair得到rank最低的merge，然后合并，直到没有可merge的
        4. 把最终的bytes token查bytes_to_id
        '''

        tokens:list[int] = []

        matches = re.finditer(PAT, text_segment)
        for match in matches:
            match_bytes = match.group().encode('utf-8')
            pretoken = tuple(bytes([byte]) for byte in match_bytes)

            while True:
                min_rank = None
                min_pair = None
                for i in range(len(pretoken) - 1):
                    pair = (pretoken[i], pretoken[i + 1])
                    rank = self.merge_ranks.get(pair)
                    if rank is None:
                        continue

                    if min_rank is None or (rank < min_rank):
                        min_rank = rank
                        min_pair = pair
                    
                if min_rank is None:
                    break

                new_pretoken_list = []
                i = 0
                while i < len(pretoken):
                    if i + 1 < len(pretoken) and ((pretoken[i], pretoken[i + 1]) == min_pair):
                        new_pretoken_list.append(min_pair[0] + min_pair[1])
                        i += 2
                    else:
                        new_pretoken_list.append(pretoken[i])
                        i += 1
                pretoken = tuple(new_pretoken_list)

            for token_bytes in pretoken:
                token_id = self.bytes_to_id[token_bytes]
                tokens.append(token_id)

        return tokens