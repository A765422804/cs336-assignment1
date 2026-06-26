from tokenizer import Tokenizer
import time

def ex1_2_3():
    print('ex1')

    chunk_size = 4096
    buffer = ''

    # 从 tiny story 采样 10 个文档
    tiny_story_doc = []
    with open('data/TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            buffer += chunk

            parts = buffer.split('<|endoftext|>')
            for part in parts[:-1]:
                tiny_story_doc.append(part)
                if len(tiny_story_doc) == 10:
                    break
            
            if len(tiny_story_doc) == 10:
                break
            
            buffer = parts[-1]

    buffer = ''
    # 从 owt 采样 10 个文档
    owt_doc = []
    with open('data/owt_train.txt', 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            buffer += chunk

            parts = buffer.split('<|endoftext|>')
            for part in parts[:-1]:
                owt_doc.append(part)
                if len(owt_doc) == 10:
                    break
            
            if len(owt_doc) == 10:
                break
            
            buffer = parts[-1]

    # 创建对应的tokenizer
    tinystories_tokenizer = Tokenizer.from_files('artifacts/tiny_stories_bpe_vocab.pkl', 'artifacts/tiny_stories_bpe_merges.pkl', ["<|endoftext|>"])
    owt_tokenizer = Tokenizer.from_files('artifacts/owt_bpe_vocab.pkl', 'artifacts/owt_bpe_merges.pkl', ["<|endoftext|>"])

    # 计算前者com ratio
    tiny_story_com_ratio = sum(len(tiny_story.encode('utf-8')) for tiny_story in tiny_story_doc) / sum(
        len(tinystories_tokenizer.encode(tiny_story)) for tiny_story in tiny_story_doc
    )
    print('tiny_stories_com_ratio = ', tiny_story_com_ratio)

    # 计算后者com ratio
    owt_com_ratio = sum(len(owt.encode('utf-8')) for owt in owt_doc) / sum(
        len(owt_tokenizer.encode(owt)) for owt in owt_doc
    )
    print('owt_com_ratio = ', owt_com_ratio)

    print('ex2')

    # 计算tiny_story_tokenizer 去 tokenize owt sample
    owt_com_ratio_on_tiny_story_tokenizer = sum(len(owt.encode('utf-8')) for owt in owt_doc) / sum(
        len(tinystories_tokenizer.encode(owt)) for owt in owt_doc
    )
    print('owt on tiny tokenizer = ', owt_com_ratio_on_tiny_story_tokenizer)

    print('ex3')

    # 计算两个doc的总长度
    tiny_doc_bytes = sum(len(doc.encode('utf-8')) for doc in tiny_story_doc)
    owt_doc_bytes = sum(len(doc.encode('utf-8')) for doc in owt_doc)
    pile_bytes = 825 * 10 ** 9

    # 记录tiny的处理时间
    start_time = time.perf_counter()
    num_tokens = sum(1 for _ in tinystories_tokenizer.encode_iterable(tiny_story_doc))
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    bytes_per_second = tiny_doc_bytes / elapsed
    print('tiny throughput = ', bytes_per_second)
    pile_seconds = pile_bytes / bytes_per_second
    pile_hours = pile_seconds / 3600
    print('tiny pile estimated hours = ', pile_hours)

    # 记录owt的处理时间
    start_time = time.perf_counter()
    num_tokens = sum(1 for _ in owt_tokenizer.encode_iterable(owt_doc))
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    bytes_per_second = owt_doc_bytes / elapsed
    print('owt throughput = ', bytes_per_second)   
    pile_seconds = pile_bytes / bytes_per_second
    pile_hours = pile_seconds / 3600
    print('owt pile estimated hours = ', pile_hours)


if __name__ == '__main__':
    ex1_2_3()
