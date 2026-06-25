import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from dataclasses import dataclass
import heapq

# pre-tokenization 使用的正则匹配项
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@dataclass(frozen=True)
class HeapItem:
    freq: int
    pair: tuple[bytes, bytes]

    def __lt__(self, other: 'HeapItem')->bool:
        if self.freq == other.freq:
            return self.pair > other.pair
        
        return self.freq > other.freq

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

def count_pretokens_for_chunk(input_path: str, start:int, end:int, special_tokens:list[str])->dict[tuple[bytes, ...], int]:
    """
    1. 子进程调用函数，处理当前进程的chunk，并且返回统计出来的pretokens
    2. 读取当前chunk的bytes，将其转为字符串
    3. 调用count_pretokens去获取当前chunk的pretoken
    """

    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode('utf-8')
        chunk_pretoken_counts = count_pretokens(chunk_text, special_tokens)

        return chunk_pretoken_counts

def merge_pretoken_counts(chunk_pretoken_counts_list: list[dict[tuple[bytes, ...], int]])->dict[tuple[bytes, ...], int]:
    """
    1. 主进程合并函数，合并所有的chunk_pretoken_counts
    """
    total_pretoken_counts: dict[tuple[bytes, ...], int] = {}
    for chunk_pretoken_counts in chunk_pretoken_counts_list:
        for bytes_tuple, count in chunk_pretoken_counts.items():
            total_pretoken_counts[bytes_tuple] = total_pretoken_counts.get(bytes_tuple, 0) + count

    return total_pretoken_counts

def build_pair_indices(pretoken_counts: dict[tuple[bytes, ...], int])->tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    1. 输入pretoken counts
    2. 统计每个pretoken内部的token-pair及其频率
    3. 统计每个token-pair对应的pretoken的集合
    """

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for pretoken, count in pretoken_counts.items():
        if len(pretoken) >= 2:
            for i in range(len(pretoken) - 1):
                cur_pair = (pretoken[i], pretoken[i + 1])
                pair_counts[cur_pair] = pair_counts.get(cur_pair, 0) + count
                pair_to_pretokens.setdefault(cur_pair, set()).add(pretoken)

    return pair_counts, pair_to_pretokens

def remove_pretoken_contribution_(old_pretoken: tuple[bytes, ...], old_pretoken_counts: int, pair_counts:dict[tuple[bytes, bytes], int], pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]], changed_pairs_set:set[tuple[bytes, bytes]]):
    """
    对于即将删除的old pretoken，移除它在pair_counts和pair_to_pretokens里面的贡献
    """
    seen_pairs: set[tuple[bytes, bytes]] = set()

    for i in range(len(old_pretoken) - 1):
        cur_token_pair = (old_pretoken[i], old_pretoken[i + 1])
        pair_counts[cur_token_pair] -= old_pretoken_counts
        seen_pairs.add(cur_token_pair)

    for seen_pair in seen_pairs:
        changed_pairs_set.add(seen_pair)
        pair_to_pretokens[seen_pair].remove(old_pretoken)
        if pair_counts[seen_pair] == 0:
            pair_counts.pop(seen_pair)
            pair_to_pretokens.pop(seen_pair)

def merge_pair_in_single_pretoken(pretoken: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes])->tuple[bytes, ...]:
    """
    遍历当前的pretoken，合并指定项
    """
    new_pretoken = []

    i = 0 
    while i < len(pretoken):
        if i + 1 < len(pretoken) and ((pretoken[i], pretoken[i + 1]) == pair_to_merge):
            new_pretoken.append(pair_to_merge[0] + pair_to_merge[1])
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1
    merged_pretoken = tuple(new_pretoken)

    return merged_pretoken

def add_pretoken_contribution_(new_pretoken: tuple[bytes, ...], added_pretoken_counts: int, pair_counts:dict[tuple[bytes, bytes], int], pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],changed_pairs_set:set[tuple[bytes, bytes]]):
    """
    对于新的pretoken，增加它在pair_counts和pair_to_pretokens里面的贡献
    """   
    seen_pairs : set[tuple[bytes, bytes]] = set()

    for i in range(len(new_pretoken) - 1):
        cur_token_pair = (new_pretoken[i], new_pretoken[i + 1])
        pair_counts[cur_token_pair] = pair_counts.get(cur_token_pair, 0) + added_pretoken_counts
        seen_pairs.add(cur_token_pair)

    for seen_pair in seen_pairs:
        changed_pairs_set.add(seen_pair)
        pair_to_pretokens.setdefault(seen_pair, set()).add(new_pretoken)
    
def merge_pair_incrementally_(pretoken_counts: dict[tuple[bytes, ...], int], best_pair: tuple[bytes, bytes], pair_counts: dict[tuple[bytes, bytes], int], pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]], pair_heap:list[HeapItem]):
    """
    同时更新pretoken counts， pair counts, pair to pretokens
    1. 找到当前best pair对应的pretokens并复制
    2. 对每个old pretoken：
        2.1. 记录remove和add中修改过的pair counts的pair信息
        2.2. 移除对pair counts和pair to pretokens的贡献
        2.3. 删除这个old pretoken从pretoken counts里
        2.4. 合并新的到pretoken count
        2.5. 把新的的贡献加进pair counts和pair to pretokens
    3. 把新的pair对象放进pair_heap
    """
    affected_pretokens = (pair_to_pretokens[best_pair]).copy()

    # 记录被修改过的pair
    changed_pairs_set: set[tuple[bytes, bytes]] = set()

    for old_pretoken in affected_pretokens:
        assert(old_pretoken in pretoken_counts)

        # 移除这个pretoken对pair counts和pair_to_pretokens的贡献
        old_pretoken_counts = pretoken_counts[old_pretoken]
        remove_pretoken_contribution_(old_pretoken, old_pretoken_counts, pair_counts, pair_to_pretokens, changed_pairs_set)

        # 在pretoken counts中删除这个pretoken
        pretoken_counts.pop(old_pretoken)

        # 构造新的pretoken
        new_pretoken = merge_pair_in_single_pretoken(old_pretoken, best_pair)
        pretoken_counts[new_pretoken] = pretoken_counts.get(new_pretoken, 0) + old_pretoken_counts

        # 增加新的pretoken对pair counts和pair_to_pretokens的贡献
        add_pretoken_contribution_(new_pretoken, old_pretoken_counts, pair_counts, pair_to_pretokens, changed_pairs_set)

    # 将修改过的pair对象放进pair heap
    for changed_pair in changed_pairs_set:
        if changed_pair in pair_counts:
            heapq.heappush(pair_heap, HeapItem(pair_counts[changed_pair], changed_pair))


def select_best_pair_from_heap(pair_heap: list[HeapItem], pair_counts: dict[tuple[bytes, bytes], int])->tuple[bytes, bytes]:
    """
    找到当前tokens pair counts里面出现频率最高，字典序最大的合法pair
    """
    while pair_heap:
        top = heapq.heappop(pair_heap)
        if top.pair in pair_counts and top.freq == pair_counts[top.pair]:
            return top.pair
        
    assert False

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    """
    1. 先把给定文本分成若干大小不一的chunk，然后对每个chunk可以并行处理了。
    2. 预处理每个chunk，执行pre-tokenizaion
    3. 汇总所有的chunk的pre-tokenization的字节tuple的频率统计结果
    4. 创建初始词汇表和merges
    5. 初始化pretoken内部相邻token pair的出现个数，以及每个token pair对应的pretoken集合，并基于token pair count建堆
    6. 开始 merge loop
    6.1. 计算出现频率最高的token pair
    6.2. 把出现频率最高的token pair合并，并且记录merge结果，并且写入vocab，更新pretoken和pair count基于pair_to_pretoken，同时更新pair_heap
    6.3. 循环以上两步直到达到最大的次数
    """

    with open(input_path, 'rb') as f:
        # 分 chunk
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 并行处理每个chunk然后合并
    tasks = []
    for start,end in zip(boundaries[:-1], boundaries[1:]):
        task = (input_path, start, end, special_tokens)
        tasks.append(task)
    
    with Pool(processes=num_processes) as pool:
        chunk_pretoken_counts_list = pool.starmap(count_pretokens_for_chunk, tasks)

    total_pretoken_counts = merge_pretoken_counts(chunk_pretoken_counts_list)

    # 创建初始词汇表
    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 初始化 token-pair counts 和对应的 token-pair 到 pretoken set 的映射
    token_pair_counts, pair_to_pretokens = build_pair_indices(total_pretoken_counts)

    # 初始化pair_heap
    pair_heap = []
    for pair, count in token_pair_counts.items():
        pair_heap.append(HeapItem(count, pair))
    heapq.heapify(pair_heap)

    while len(vocab) < vocab_size:
        if len(token_pair_counts) == 0:
            break

        # 计算出现频率最高的pair
        best_pair = select_best_pair_from_heap(pair_heap, token_pair_counts)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        merge_pair_incrementally_(total_pretoken_counts, best_pair, token_pair_counts, pair_to_pretokens, pair_heap)

    return vocab, merges

if __name__ == '__main__':
    # do test
    pass