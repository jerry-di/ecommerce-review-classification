"""
model_train_mtl - 多任务训练（情感二分类 + 评价维度六分类）

结构:
    共享 BERT encoder
        ├── 情感头: 2 分类 (好评/差评)     <- 京东评论 CSV (约 4.5 万条)
        └── 维度头: 6 分类 (物流/质量/...)  <- DeepSeek 伪标注数据

特性:
    - 早停: 维度验证准确率连续 patience 个 epoch 未提升则停止
    - 只保存最优模型: 仅当验证准确率创新高时保存 checkpoint

用法:
    python src/model_train_mtl.py --epochs 8 --patience 2
    python src/model_train_mtl.py --resume --epochs 5 --patience 2
"""
import argparse
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from modelscope import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.utils import get_corpus


class MultiTaskBERT(nn.Module):
    """共享 encoder + 双头"""

    def __init__(self, encoder_model, num_sent=2, num_asp=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_model)
        hidden = self.bert.config.hidden_size
        self.sent_head = nn.Linear(hidden, num_sent)
        self.asp_head = nn.Linear(hidden, num_asp)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.sent_head(pooled), self.asp_head(pooled)


def compute_class_weights(corpus, num_classes):
    """按反频率计算类别权重，缓解类别不平衡"""
    counts = np.bincount([label for _, label in corpus], minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def tokenize_corpus(corpus, tokenizer, max_len=32):
    """批量预分词整个语料，返回 (input_ids, attention_mask, labels)"""
    texts = [doc for doc, _ in corpus]
    labels = torch.tensor([label for _, label in corpus], dtype=torch.int64)
    enc = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt',
    )
    return enc['input_ids'], enc['attention_mask'], labels


def make_loader(data, batch_size, shuffle=True, drop_last=True):
    return DataLoader(
        TensorDataset(*data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )


def build_model(resume: bool) -> MultiTaskBERT:
    """构建模型；resume=True 时从上次保存的 checkpoint 加载"""
    if resume:
        encoder = str(Config.ft_bert_file)
        model = MultiTaskBERT(encoder, num_sent=2, num_asp=Config.num_labels)
        heads_path = str(Config.ft_bert_file) + '_mtl_heads.pt'
        ckpt = torch.load(heads_path, map_location='cpu')
        model.sent_head.load_state_dict(ckpt['sent_head'])
        model.asp_head.load_state_dict(ckpt['asp_head'])
        print('已加载上次训练的模型 checkpoint')
    else:
        model = MultiTaskBERT(Config.pretrained_model, num_sent=2, num_asp=Config.num_labels)
    return model


def save_model(model, tokenizer):
    """保存模型（encoder + 双头 + tokenizer）"""
    model.bert.save_pretrained(Config.ft_bert_file)
    torch.save(
        {
            'sent_head': model.sent_head.state_dict(),
            'asp_head': model.asp_head.state_dict(),
        },
        str(Config.ft_bert_file) + '_mtl_heads.pt',
    )
    tokenizer.save_pretrained(Config.ft_bert_file)


@torch.inference_mode()
def evaluate(model, data, device, head=0, batch_size=256):
    """评估指定头（0=情感, 1=维度）的准确率"""
    model.eval()
    loader = DataLoader(TensorDataset(*data), batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        sent_logits, asp_logits = model(input_ids, attention_mask)
        logits = sent_logits if head == 0 else asp_logits
        y_true.extend(labels.numpy())
        y_pred.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    return accuracy_score(y_true, y_pred)


def train():
    parser = argparse.ArgumentParser(description='多任务 BERT 训练（带早停 + 保存最优）')
    parser.add_argument('--epochs', type=int, default=8, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=2, help='早停耐心（连续多少个 epoch 不提升则停止）')
    parser.add_argument('--resume', action='store_true', help='从上次 checkpoint 续训')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f'设备: {device}')

    tokenizer = AutoTokenizer.from_pretrained(Config.pretrained_model)
    model = build_model(args.resume)
    model.to(device)

    sent_train = get_corpus(Config.sentiment_train_file)
    sent_valid = get_corpus(Config.sentiment_valid_file)
    asp_train = get_corpus(Config.train_raw_file)
    asp_valid = get_corpus(Config.valid_raw_file)
    print(f'情感: 训练 {len(sent_train)} / 验证 {len(sent_valid)}')
    print(f'维度: 训练 {len(asp_train)} / 验证 {len(asp_valid)}')

    print('预分词中...')
    sent_train_data = tokenize_corpus(sent_train, tokenizer)
    sent_valid_data = tokenize_corpus(sent_valid, tokenizer)
    asp_train_data = tokenize_corpus(asp_train, tokenizer)
    asp_valid_data = tokenize_corpus(asp_valid, tokenizer)
    print('预分词完成')

    sent_loader = make_loader(sent_train_data, batch_size=256, shuffle=True)
    asp_loader = make_loader(asp_train_data, batch_size=64, shuffle=True)

    ce_sent = nn.CrossEntropyLoss()
    asp_weights = compute_class_weights(asp_train, Config.num_labels).to(device)
    ce_asp = nn.CrossEntropyLoss(weight=asp_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    LAMBDA = 1.0

    best_asp = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        asp_iter = cycle(asp_loader)
        total_loss = 0.0
        steps = 0

        for sent_ids, sent_mask, sent_labels in sent_loader:
            asp_ids, asp_mask, asp_labels = next(asp_iter)
            sent_ids = sent_ids.to(device)
            sent_mask = sent_mask.to(device)
            sent_labels = sent_labels.to(device)
            asp_ids = asp_ids.to(device)
            asp_mask = asp_mask.to(device)
            asp_labels = asp_labels.to(device)

            optimizer.zero_grad()
            sent_logits, _ = model(sent_ids, sent_mask)
            loss_sent = ce_sent(sent_logits, sent_labels)
            _, asp_logits = model(asp_ids, asp_mask)
            loss_asp = ce_asp(asp_logits, asp_labels)
            loss = loss_sent + LAMBDA * loss_asp

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f'Epoch[{epoch}/{args.epochs}] 平均 loss: {total_loss / steps:.4f}')

        sent_acc = evaluate(model, sent_valid_data, device, head=0)
        asp_acc = evaluate(model, asp_valid_data, device, head=1)
        print(f'  [验证] 情感准确率: {sent_acc:.2%} | 维度准确率: {asp_acc:.2%}')

        if asp_acc > best_asp:
            best_asp = asp_acc
            best_epoch = epoch
            no_improve = 0
            save_model(model, tokenizer)
            print(f'  [保存] 维度准确率创新高，已保存最优模型')
        else:
            no_improve += 1
            print(f'  [未提升] 已连续 {no_improve} 个 epoch 未超过 {best_asp:.2%}')
            if no_improve >= args.patience:
                print(f'早停: 连续 {args.patience} 个 epoch 未提升，停止训练')
                break

    print(f'训练结束 | 最优维度准确率: {best_asp:.2%} (第 {best_epoch} 个 epoch)')
    print(f'最优模型已保存到 {Config.ft_bert_file}')


if __name__ == '__main__':
    train()
