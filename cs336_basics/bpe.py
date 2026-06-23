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


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    """
    1. 先把给定文本分成若干大小不一的chunk，然后对每个chunk可以并行处理了。
    2. 预处理每个chunk，执行pre-tokenizaion
    3. 汇总所有的chunk的pre-tokenization的字节tuple的频率统计结果
    4. TODO：随着代码完成继续补充
    6. 
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

    # TODO:验证“分 chunk 后汇总”与“整个文件一次性调用 count_pretokens”得到的字典完全相等。用一个较小、包含多个 <|endoftext|> 的文本测试，这能验证分块没有改变预分词结果。

if __name__ == '__main__':
    # do test

    small_text = "<|endoftext|>abc<|endoftext|><|endoftext|>def<|endoftext|>"

    pretoken_dict = count_pretokens(small_text, ["<|endoftext|>"])

    for key, value in pretoken_dict.items():
        print(key, value)
        print(b"".join(key))

    