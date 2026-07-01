'''
画图
输入jsonl
输出step-X图
'''

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_metric(log_paths: list[str], labels: list[str], x_key:str, y_key:str, out_dir:str):
    '''
    实际绘图函数
    1. 创建画布
    2. 遍历每个path根据x和ykey读取对应元素
    3. 设置画布的label
    4. save
    '''

    plt.figure(figsize=(8, 5))

    for log_path, label in zip(log_paths,labels):
        x = []
        y = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if x_key in record and y_key in record:
                    x.append(record[x_key])
                    y.append(record[y_key])

        plt.plot(x, y, label=label)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f'{y_key} vs {x_key}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(Path(out_dir) / f'{y_key}_vs_{x_key}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logs',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True
    )
    args = parser.parse_args()

    assert(len(args.logs) == len(args.labels))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    plot_metric(args.logs, args.labels, 'step', 'lr', args.out_dir)
    plot_metric(args.logs, args.labels, 'step', 'train_loss', args.out_dir)
    plot_metric(args.logs, args.labels, 'step', 'val_loss', args.out_dir)

if __name__ == '__main__':
    main()
