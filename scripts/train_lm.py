import argparse
import yaml
import random
import numpy as np
import torch
import time
from tqdm import tqdm
import json
from pathlib import Path
from cs336_basics.nn import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, gradient_clipping
from cs336_basics.training import load_checkpoint
from cs336_basics.optimizer import cos_learning_rate_schedule
from cs336_basics.data import get_batch

def set_seed(seed: int):
    '''
    设置python numpy pytorch(CPU|GPU)随机数
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def training_loop(config: dict):
    '''
    训练循环：
    1. 设置随机数种子
    2. 选择device
    3. 加载数据
    4. 创建模型放进device
    5. 创建optimizer放进device
    6. 如果有resume checkpoint，加载checkpoint以及iteration
    7. 实际开始训练:
        1. set lr
        2. get batch
        3. zero_grad
        4. forward
        5. loss
        6. backward
        7. clip grads
        8. optimizer.step
        9. log / eval / checkpoint
    '''

    # 设置随机数种子
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # 选择设备
    device = config['training'].get('device', 'cpu')

    # 加载数据
    train_data = np.load(config['data']['train_path'], mmap_mode='r')
    val_data = np.load(config['data']['val_path'], mmap_mode='r')

    # 创建模型
    context_length = config['model']['context_length']
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        context_length=context_length,
        num_layers=config['model']['num_layers'],
        d_model = config['model']['d_model'],
        num_heads= config['model']['num_heads'],
        d_ff = config['model']['d_ff'],
        rope_theta=config['model']['rope_theta'],
    )
    model = model.to(device)

    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        betas=tuple(config['optimizer']['betas']),
        eps=config['optimizer']['eps'],
        weight_decay=config['optimizer']['weight_decay']
    )

    # 加载checkpoint
    start_iter = 0
    resume_path = config['checkpoint'].get('resume_path', None)
    if resume_path is not None:
        start_iter = load_checkpoint(resume_path, model, optimizer)

    # 加载config参数
    log_interval = config['logging']['log_interval']
    eval_interval = config['logging']['eval_interval']
    batch_size = config['training']['batch_size']

    # 正式开始训练
    max_iters = config['training']['max_iters']
    start_time = time.perf_counter()
    log_path = config['logging']['log_path']
    Path(log_path).parent.mkdir(exist_ok=True)
    pbar = tqdm(range(start_iter, max_iters))
    with open(log_path, 'a', encoding='utf-8') as log_file:
        for t in pbar:
            # set lr
            lr = cos_learning_rate_schedule(
                t,
                config['lr_schedule']['alpha_max'],
                config['lr_schedule']['alpha_min'],
                config['lr_schedule']['warmup_iters'],
                config['lr_schedule']['cosine_iters']
            )
            for group in optimizer.param_groups:
                group['lr'] = lr

            # get batch
            inputs, targets = get_batch(train_data, batch_size ,context_length, device)

            # zero_grad
            optimizer.zero_grad()

            # forward
            logits = model(inputs)

            # loss
            loss = cross_entropy(logits, targets)

            # backward
            loss.backward()

            # clip grads
            grad_clip = config['training'].get('grad_clip', None)
            if grad_clip:
                gradient_clipping(model.parameters(), grad_clip)

            # optimizer.step
            optimizer.step()

            # log / eval / checkpoint
            pbar.set_postfix(loss=loss.item(), lr=lr)

            need_train_log = (t + 1) % log_interval == 0
            need_eval_log = (t + 1) % eval_interval == 0

            if need_train_log or need_eval_log:
                elapsed = time.perf_counter() - start_time
                record = {
                    'step' : t + 1,
                    'time_sec': elapsed,
                    'lr' : lr
                }

                if need_train_log:
                    record['train_loss'] = loss.item()

                if need_eval_log:
                    eval_iters = config['logging']['eval_iters']
                    val_loss = 0
                    model.eval()
                    with torch.no_grad():
                        for _ in range(eval_iters):
                            eval_inputs, eval_targets = get_batch(val_data, batch_size, context_length, device)
                            eval_logits = model(eval_inputs)
                            val_loss += cross_entropy(eval_logits, eval_targets).item()
                        val_loss /= eval_iters
                    model.train()

                    record['val_loss'] = val_loss

                json.dump(record, log_file)
                log_file.write('\n')
                log_file.flush()
            

def main():
    '''
    training loop 主脚本：
    1. 读取config
    2. 调用训练流程
    '''

    # 读取config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config file'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print('load config finish')

    # 调用训练流程
    training_loop(config)

    print('training loop finish')

if __name__ == '__main__':
    main()