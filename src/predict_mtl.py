"""
predict_mtl - 用训练好的多任务模型做推理（支持 DeepSeek 低置信度兜底）

策略:
    本地模型先预测；当维度置信度低于阈值时，调用 DeepSeek 重新判断维度。
    这样简单样本走本地模型（快、免费），疑难样本走 DeepSeek（更准）。

用法:
    python src/predict_mtl.py "物流很快第二天就到了"
    python src/predict_mtl.py "耳机音质差" "客服态度好" --threshold 0.7
    python src/predict_mtl.py "..." --no-fallback          # 禁用兜底
"""
import argparse
import os
import sys

import torch
from modelscope import AutoTokenizer

from src.config import Config
from src.model_train_mtl import MultiTaskBERT
from src.pseudo_label import classify_one, load_class_labels


def load():
    """加载保存的模型（encoder + 双头）"""
    tokenizer = AutoTokenizer.from_pretrained(str(Config.ft_bert_file))
    model = MultiTaskBERT(str(Config.ft_bert_file), num_sent=2, num_asp=Config.num_labels)
    ckpt = torch.load(str(Config.ft_bert_file) + '_mtl_heads.pt', map_location='cpu')
    model.sent_head.load_state_dict(ckpt['sent_head'])
    model.asp_head.load_state_dict(ckpt['asp_head'])
    model.eval()
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description='多任务文本分类推理')
    parser.add_argument('texts', nargs='+', help='要分类的评论文本')
    parser.add_argument('--threshold', type=float, default=0.8, help='维度置信度阈值，低于则用 DeepSeek 兜底')
    parser.add_argument('--no-fallback', action='store_true', help='禁用 DeepSeek 兜底')
    args = parser.parse_args()

    tokenizer, model = load()
    labels = load_class_labels()
    sent_labels = ['差评', '好评']

    enc = tokenizer(args.texts, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
    with torch.inference_mode():
        sent_logits, asp_logits = model(enc['input_ids'], enc['attention_mask'])
        sent_conf, sent_pred = torch.softmax(sent_logits, dim=-1).max(dim=-1)
        asp_conf, asp_pred = torch.softmax(asp_logits, dim=-1).max(dim=-1)

    headers = None
    if not args.no_fallback:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key:
            headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        else:
            print('提示: 未设置 DEEPSEEK_API_KEY，兜底不可用，将只用本地模型\n')

    for i, text in enumerate(args.texts):
        sent = sent_labels[sent_pred[i].item()]
        aspect = labels[asp_pred[i].item()]
        sc = sent_conf[i].item()
        ac = asp_conf[i].item()

        used = False
        if headers and ac < args.threshold:
            try:
                aspect = classify_one(text, labels, headers)
                used = True
            except Exception as exc:  # noqa: BLE001
                print(f'  [兜底失败，改用本地结果] {exc}')

        tag = ' [DeepSeek 兜底]' if used else ''
        print(f'{text}')
        print(f'  情感: {sent} ({sc:.0%}) | 维度: {aspect} ({ac:.0%}){tag}')


if __name__ == '__main__':
    main()
