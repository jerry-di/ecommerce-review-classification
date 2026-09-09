"""
pseudo_label_pack - 定向补标「包装」类（候选来自京东 CSV 语料）

用法:
    python src/pseudo_label_pack.py

产物:
    data/raw/eprstmt_aspect_boost2.txt
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import Config
from src.pseudo_label import load_class_labels, classify_one

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / 'data' / 'raw'
OUT = RAW / 'eprstmt_aspect_boost2.txt'

KWS = ['包装', '盒子', '外包装', '包装盒', '礼盒', '破损', '压扁', '压坏', '拆箱', '拆开', '封条', '纸箱']


def load_labeled() -> set[str]:
    texts = set()
    for f in ['eprstmt_aspect_train.txt', 'eprstmt_aspect_valid.txt']:
        for line in open(RAW / f, encoding='utf-8'):
            texts.add(line.split('\t', 1)[0])
    return texts


def main():
    headers = {
        'Authorization': f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        'Content-Type': 'application/json',
    }
    labels = load_class_labels()
    label2id = {name: i for i, name in enumerate(labels)}

    labeled = load_labeled()
    candidates = set()
    for f in ['jd_sentiment_train.txt', 'jd_sentiment_valid.txt']:
        for line in open(RAW / f, encoding='utf-8'):
            text = line.split('\t', 1)[0]
            if text and text not in labeled and any(k in text for k in KWS):
                candidates.add(text)
    candidates = list(candidates)
    print(f'候选: {len(candidates)} 条')

    results = {}
    done = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(os.getenv('DEEPSEEK_WORKERS', '20'))) as pool:
        future_to_text = {
            pool.submit(classify_one, text, labels, headers): text
            for text in candidates
        }
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                results[text] = label2id[future.result()]
            except Exception as exc:  # noqa: BLE001
                print(f'[失败] {text[:30]}... -> {exc}')
            done += 1
            if done % 200 == 0:
                elapsed = time.perf_counter() - start
                print(f'进度: {done}/{len(candidates)} ({elapsed:.0f}s)')

    with open(OUT, 'w', encoding='utf-8') as f:
        for text, label in results.items():
            f.write(f'{text}\t{label}\n')

    print(f'完成: 新增 {len(results)} 条, 已写入 {OUT}')


if __name__ == '__main__':
    main()
