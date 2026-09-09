"""
pseudo_label - 用 DeepSeek 给无标签电商评论批量打多分类(评价维度)标签

思路:
1. 从 1.9 万条无标签评论中随机抽取 N 条（默认 5000）
2. 并发调用 DeepSeek 判断评论主要涉及的评价维度
3. 写入 eprstmt_aspect_labeled.txt（随后再切分成训练/验证集）

类别（6 类评价维度）:
    物流 / 质量 / 价格 / 客服 / 包装 / 其它

依赖:
    requests（已在 requirements.txt）

环境变量:
    DEEPSEEK_API_KEY   必填，DeepSeek 平台 API Key
    DEEPSEEK_BASE_URL  可选，默认 https://api.deepseek.com
    DEEPSEEK_MODEL     可选，默认 deepseek-chat
    DEEPSEEK_WORKERS   可选，并发线程数，默认 20

用法:
    python src/pseudo_label.py --n 5000
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from src.config import Config


def _load_dotenv():
    """从项目根目录 .env 加载环境变量（不覆盖已存在的变量）"""
    env_file = Path(__file__).resolve().parent.parent / '.env'
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
MAX_WORKERS = int(os.getenv('DEEPSEEK_WORKERS', '20'))

# 各类别定义，用于 prompt 说明
CATEGORY_DEFS = {
    '物流': '配送速度、发货、快递、收货、物流服务',
    '质量': '商品本身的好坏、做工、耐用性、功能是否正常、真假',
    '价格': '性价比、价格高低、降价、优惠、是否划算',
    '客服': '售前咨询、售后服务、客服/商家态度、退换货处理',
    '包装': '外包装、包装是否精美、有无破损',
    '其它': '不属于以上任何一类，或只是整体评价/使用感受',
}

# 手写 few-shot 示例（每类 2 条，正反各一）
FEW_SHOT = [
    ('物流很快，第二天就到了，很满意', '物流'),
    ('快递太慢了，等了一个星期才收到', '物流'),
    ('耳机音质很差，用了两天就坏了', '质量'),
    ('做工很精致，用料扎实，用起来很放心', '质量'),
    ('这个价格买到真的很划算，性价比高', '价格'),
    ('太贵了，完全不值这个价', '价格'),
    ('客服态度很好，耐心解答了我的问题', '客服'),
    ('售后很差，出了问题根本没人管', '客服'),
    ('包装很精美，送人也很有面子', '包装'),
    ('外包装破损，盒子都被压扁了', '包装'),
    ('和描述的一致，整体挺满意的', '其它'),
    ('买来送人的，对方很喜欢', '其它'),
]


def load_class_labels() -> list[str]:
    """读取类别标签列表（顺序与 label 编号一致）"""
    with open(Config.class_file, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt(text: str, labels: list[str]) -> str:
    """构造多分类 prompt"""
    category_lines = '\n'.join(
        f'- {label}：{CATEGORY_DEFS[label]}' for label in labels
    )
    few_shot_lines = '\n\n'.join(
        f'评论: {t}\n标签: {l}' for t, l in FEW_SHOT
    )
    return (
        '你是电商评论归类助手，判断这条评论主要在说哪个方面，从以下类别中选一个：\n'
        f'{category_lines}\n\n'
        '如果一条评论同时提到多个方面，选最主要、最强调的那一个。\n\n'
        f'示例：\n{few_shot_lines}\n\n'
        '请判断下面这条评论，只输出一个 JSON 对象，格式为 {"label": "<类别>"}，'
        'label 必须是上述类别之一，不要输出其它内容：\n'
        f'评论: {text}'
    )


def classify_one(text: str, labels: list[str], headers: dict) -> str:
    """调用 DeepSeek 判定单条评论的评价维度，返回类别名"""
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': build_prompt(text, labels)}],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
        'stream': False,
    }

    last_err = None
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
            if label not in labels:
                raise ValueError(f'非法标签: {label!r}')
            return label
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f'三次重试后仍失败: {last_err}')


def main():
    parser = argparse.ArgumentParser(description='DeepSeek 伪标注电商评论（评价维度）')
    parser.add_argument('--n', type=int, default=5000, help='伪标注的样本数量')
    args = parser.parse_args()

    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        sys.exit('错误: 请先设置环境变量 DEEPSEEK_API_KEY')

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    labels = load_class_labels()
    label2id = {name: i for i, name in enumerate(labels)}
    print(f'类别: {labels}')

    # 抽取无标签样本
    with open(Config.unlabeled_file, encoding='utf-8') as f:
        unlabeled = [line.strip() for line in f if line.strip()]
    if args.n > len(unlabeled):
        args.n = len(unlabeled)
    samples = random.Random(42).sample(unlabeled, args.n)

    print(f'开始伪标注: 共 {args.n} 条, 并发 {MAX_WORKERS}, 模型 {MODEL}')

    pseudo = {}
    done = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_text = {
            pool.submit(classify_one, text, labels, headers): text
            for text in samples
        }
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                pseudo[text] = label2id[future.result()]
            except Exception as exc:  # noqa: BLE001
                print(f'[失败] {text[:40]}... -> {exc}')
            done += 1
            if done % 200 == 0:
                elapsed = time.perf_counter() - start
                print(f'进度: {done}/{args.n} ({elapsed:.0f}s)')

    with open(Config.pseudo_labeled_file, 'w', encoding='utf-8') as f:
        for text, label in pseudo.items():
            f.write(f'{text}\t{label}\n')

    print(f'完成: 成功 {len(pseudo)} 条, 已写入 {Config.pseudo_labeled_file}')


if __name__ == '__main__':
    main()
