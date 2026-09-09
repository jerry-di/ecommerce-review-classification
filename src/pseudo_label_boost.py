"""
pseudo_label_boost - 定向补标稀有维度

用关键词从剩余无标签语料中筛选候选评论，交给 DeepSeek 打评价维度标签，
用于补足物流 / 价格 / 客服 / 包装等稀有类别的样本量。

用法:
    python src/pseudo_label_boost.py

产物:
    data/raw/eprstmt_aspect_boost.txt  （本轮新增的伪标签）
"""
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import Config
from src.pseudo_label import load_class_labels, classify_one

BASE_DIR = Path(__file__).resolve().parent.parent
BOOST_OUT = BASE_DIR / 'data' / 'raw' / 'eprstmt_aspect_boost.txt'

KEYWORDS = {
    '物流': ['物流', '快递', '发货', '配送', '收货', '到货', '揽收', '签收', '送货', '派送', '顺丰', '圆通', '中通', '韵达', '申通'],
    '价格': ['价格', '性价比', '划算', '贵', '便宜', '实惠', '降价', '优惠', '促销', '打折', '不值', '亏'],
    '客服': ['客服', '售后', '态度', '咨询', '退货', '退款', '换货', '退换', '维权', '服务'],
    '包装': ['包装', '盒子', '外包装', '破损', '压扁', '压坏', '拆箱', '拆开', '封条'],
}


def load_labeled_texts() -> set[str]:
    """读取已标注文本，避免重复"""
    texts = set()
    if Config.pseudo_labeled_file.exists():
        with open(Config.pseudo_labeled_file, encoding='utf-8') as f:
            for line in f:
                texts.add(line.split('\t', 1)[0])
    return texts


def load_unlabeled() -> list[str]:
    with open(Config.unlabeled_file, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise SystemExit('错误: 请先设置环境变量 DEEPSEEK_API_KEY')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    labels = load_class_labels()
    label2id = {name: i for i, name in enumerate(labels)}

    labeled = load_labeled_texts()
    unlabeled = load_unlabeled()

    # 关键词筛选候选
    candidates = set()
    for kws in KEYWORDS.values():
        for text in unlabeled:
            if text not in labeled and text not in candidates and any(k in text for k in kws):
                candidates.add(text)
    candidates = list(candidates)
    random.Random(42).shuffle(candidates)
    print(f'候选: {len(candidates)} 条')

    pseudo = {}
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
                pseudo[text] = label2id[future.result()]
            except Exception as exc:  # noqa: BLE001
                print(f'[失败] {text[:30]}... -> {exc}')
            done += 1
            if done % 200 == 0:
                elapsed = time.perf_counter() - start
                print(f'进度: {done}/{len(candidates)} ({elapsed:.0f}s)')

    with open(BOOST_OUT, 'w', encoding='utf-8') as f:
        for text, label in pseudo.items():
            f.write(f'{text}\t{label}\n')

    print(f'完成: 新增 {len(pseudo)} 条, 已写入 {BOOST_OUT}')


if __name__ == '__main__':
    main()
