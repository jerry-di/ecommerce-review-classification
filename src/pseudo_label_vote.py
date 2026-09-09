"""
pseudo_label_vote - 多票表决重标（降低伪标签噪声）

对已标注的维度数据，让 DeepSeek 对同一条评论投 3 票（temperature=0.7 引入多样性），
取多数票作为更干净的标签；三票全不同的样本视为过噪、丢弃。

用法:
    python src/pseudo_label_vote.py

产物:
    重写 data/raw/eprstmt_aspect_train.txt / eprstmt_aspect_valid.txt（多数票标签 + 重新分层切分）
"""
import collections
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from sklearn.model_selection import train_test_split

from src.config import Config
from src.pseudo_label import build_prompt, load_class_labels

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / 'data' / 'raw'
VOTES = 3
BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')


def vote_once(text, labels, headers):
    """对单条文本投一票（temperature=0.7 引入多样性）"""
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': build_prompt(text, labels)}],
        'temperature': 0.7,
        'response_format': {'type': 'json_object'},
        'stream': False,
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                f'{BASE_URL}/chat/completions',
                json=payload,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()['choices'][0]['message']['content']
            label = json.loads(content)['label']
            if label in labels:
                return label
            raise ValueError(f'非法标签: {label!r}')
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    raise RuntimeError('投票失败')


def load_all_samples():
    """读取当前 train + valid 的样本（text -> 原标签）"""
    samples = {}
    for f in ['eprstmt_aspect_train.txt', 'eprstmt_aspect_valid.txt']:
        for line in open(RAW / f, encoding='utf-8'):
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2 and parts[0] and parts[0] not in samples:
                samples[parts[0]] = int(parts[1])
    return samples


def main():
    headers = {
        'Authorization': f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        'Content-Type': 'application/json',
    }
    labels = load_class_labels()
    label2id = {name: i for i, name in enumerate(labels)}

    samples = load_all_samples()
    texts = list(samples.keys())
    total = len(texts)
    print(f'待重标: {total} 条, 每条 {VOTES} 票')

    tasks = [(text, v) for text in texts for v in range(VOTES)]
    votes = {text: [] for text in texts}
    done = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(os.getenv('DEEPSEEK_WORKERS', '20'))) as pool:
        future_to_task = {
            pool.submit(vote_once, text, labels, headers): (text, v)
            for text, v in tasks
        }
        for future in as_completed(future_to_task):
            text, _ = future_to_task[future]
            try:
                votes[text].append(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f'[失败] {text[:30]}... -> {exc}')
            done += 1
            if done % 600 == 0:
                elapsed = time.perf_counter() - start
                print(f'进度: {done}/{len(tasks)} ({elapsed:.0f}s)')

    # 多数票
    final = {}
    agree_all = 0
    flipped = 0
    dropped = 0
    for text, vs in votes.items():
        if len(vs) < VOTES:
            dropped += 1
            continue
        counter = collections.Counter(vs)
        top_label, top_count = counter.most_common(1)[0]
        if top_count == VOTES:
            agree_all += 1
        if top_count >= 2:
            new_id = label2id[top_label]
            if new_id != samples[text]:
                flipped += 1
            final[text] = new_id
        else:
            dropped += 1  # 三票全不同

    n = len(final)
    print(f'三票全一致: {agree_all}/{total} ({agree_all/total:.1%})')
    print(f'标签被改: {flipped}/{n} ({flipped/n:.1%})')
    print(f'最终保留 {n} 条, 丢弃 {dropped} 条')

    # 重新分层切分
    texts = list(final.keys())
    y = [final[t] for t in texts]
    train_t, valid_t, train_l, valid_l = train_test_split(
        texts, y, test_size=0.15, random_state=42, stratify=y
    )
    with open(RAW / 'eprstmt_aspect_train.txt', 'w', encoding='utf-8') as f:
        for t, l in zip(train_t, train_l):
            f.write(f'{t}\t{l}\n')
    with open(RAW / 'eprstmt_aspect_valid.txt', 'w', encoding='utf-8') as f:
        for t, l in zip(valid_t, valid_l):
            f.write(f'{t}\t{l}\n')

    dist = collections.Counter(y)
    print(f'训练 {len(train_t)} / 验证 {len(valid_t)}')
    print('新分布:', {labels[i]: dist[i] for i in range(len(labels))})


if __name__ == '__main__':
    main()
