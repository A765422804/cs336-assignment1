import os
import regex as re
from typing import BinaryIO

# pre-tokenization 使用的正则匹配项
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_pretokens(chunk_text: str, special_tokens: list[str])->dict[tuple[bytes, ...], int]:
    """
    1. 按所有 special_tokens 切分文本，而不是简单替换删除，确保两侧形成不可跨越的边界。
    2. 特殊 token 本身不参与预分词和频率统计。
    3. 对每个普通文本片段分别用指定正则的 finditer 预分词。
    4. 将每个匹配出的 pre-token 编码为 UTF-8。
    5. 把每个字节整数转换为长度为 1 的 bytes 对象，组成 tuple。
    6. 统计相同 tuple 的总出现次数。
    """

    if special_tokens:
        pattern  = '|'.join([re.escape(special_token) for special_token in special_tokens])
        text_segments = re.split(pattern, chunk_text)
    else:
        text_segments = [chunk_text]

    pretoken_counts: dict[tuple[bytes, ...], int] = {}
    for segment in text_segments:
        matches = re.finditer(PAT, segment)
        for match in matches:
            match_bytes = match.group().encode('utf-8')
            bytes_tuple = tuple(bytes([byte]) for byte in match_bytes)
            pretoken_counts[bytes_tuple] = pretoken_counts.get(bytes_tuple, 0) + 1

    return pretoken_counts

def init_vocab(special_tokens: list[str])->dict[int, bytes]:
    """
    1. 创建256个基础byte token
    2. 把special_tokens追加进vocab
    4. 返回dict[int, bytes]
    """
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')

    return vocab

def count_adjacent_pairs(pretoken_counts:dict[tuple[bytes, ...], int])->dict[tuple[bytes, bytes], int]:
    """
    1. 统计输入的dict的每个pretoken内部相邻bytes的出现频率
    2. 返回bytes-pair出现频率字典
    """
    
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for pretoken,count in pretoken_counts.items():
        if len(pretoken) >=2:
            for i in range(len(pretoken) - 1):
                cur_pair = (pretoken[i], pretoken[i + 1])
                pair_counts[cur_pair] = pair_counts.get(cur_pair, 0) + count

    return pair_counts

def select_best_pair(pair_counts: dict[tuple[bytes, bytes], int])->tuple[bytes, bytes]:
    """
    1. 选择输入的dict的出现最高的bytes-pair
    2. 返回出现最高的bytes-pair
    """
    best_pair: tuple[bytes, bytes] | None = None
    best_count = -1

    for pair, count in pair_counts.items():
        if best_pair is None or count > best_count or (count == best_count and pair > best_pair):
            best_pair = pair
            best_count = count

    assert best_pair is not None
    
    return best_pair

def merge_pair_in_pretokens(pretoken_counts: dict[tuple[bytes, ...], int], pair_to_merge: tuple[bytes, bytes])->dict[tuple[bytes, ...], int]:
    """
    1. 遍历每个pretoken，把其中出现的pair合并
    2. 如果合并以后和之前的pretoken重复了，就把count加起来
    """ 
    merged_pretoken_counts: dict[tuple[bytes, ...], int] = {}
    for pretoken, count in pretoken_counts.items():
        merged_tokens = []
        pretoken_lens = len(pretoken)
        i = 0
        while i < pretoken_lens:
            if i+1 <pretoken_lens and ((pretoken[i], pretoken[i + 1]) == pair_to_merge):
                merged_tokens.append(pair_to_merge[0] + pair_to_merge[1])
                i += 2
            else:
                merged_tokens.append(pretoken[i])
                i += 1
        merged_pretoken = tuple(merged_tokens)
        merged_pretoken_counts[merged_pretoken] = merged_pretoken_counts.get(merged_pretoken, 0) + count
        
    return merged_pretoken_counts



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    """
    1. 先把给定文本分成若干大小不一的chunk，然后对每个chunk可以并行处理了。
    2. 预处理每个chunk，执行pre-tokenizaion
    3. 汇总所有的chunk的pre-tokenization的字节tuple的频率统计结果
    4. 创建初始词汇表和merges
    5. 开始 merge loop
    5.1. 计算pretoken内部相邻bytes的出现个数
    5.2. 计算出现频率最高的token pair
    5.3. 把出现频率最高的token pair合并，并且记录merge结果，并且写入vocab
    5.4. 循环以上三步直到达到最大的次数
    """

    with open(input_path, 'rb') as f:
        # 分 chunk
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # 逐 chunk 处理 TODO：改成并行的
        total_pretoken_counts: dict[tuple[bytes, ...], int] = {}
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_text = f.read(end - start).decode('utf-8')
            chunk_pretoken_counts = count_pretokens(chunk_text, special_tokens) 
            for byte_tuple, count in chunk_pretoken_counts.items():
                total_pretoken_counts[byte_tuple] = total_pretoken_counts.get(byte_tuple, 0) + count

    # 创建初始词汇表
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # 计算相邻token pair的频率
        pair_counts = count_adjacent_pairs(total_pretoken_counts)

        if len(pair_counts) == 0:
            break

        # 计算出现频率最高的pair
        best_pair = select_best_pair(pair_counts)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # 在pretoken中把出现频率最高的token pair合并
        total_pretoken_counts = merge_pair_in_pretokens(total_pretoken_counts, best_pair)

    return vocab, merges

if __name__ == '__main__':
    # do test

    vocab, merges = train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 256 + 1 + 10, ["<|endoftext|>"])

    print(len(vocab))
    print(len(merges))
    print(merges)
    print(list(vocab.items())[-10:])