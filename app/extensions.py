"""
extensions - 扩展功能

加载多任务分类模型（共享 BERT + 情感头 + 维度头），
提供带 DeepSeek 低置信度兜底的推理能力。
"""
import os
from threading import Lock

import torch
from flask import Flask
from modelscope import AutoTokenizer

from src.config import Config
from src.model_train_mtl import MultiTaskBERT
from src.pseudo_label import classify_one
from app.storage import redis_cache, vector_store


class TextClassifierExtension:
    """文本分类器单例扩展（多任务 BERT + DeepSeek 低置信度兜底）"""

    def __init__(self, app: Flask = None):
        self._model = None
        self._class_labels = None
        self._lock = Lock()
        self._infer_lock = Lock()  # 串行化 GPU 推理，保护模型与显存
        self._fallback_threshold = 0.8
        self._sent_labels = ['差评', '好评']
        self.tokenizer = AutoTokenizer.from_pretrained(str(Config.ft_bert_file))
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """挂载到 Flask 实例"""
        self._fallback_threshold = app.config.get('FALLBACK_THRESHOLD', 0.8)
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        app.extensions['text_classifier'] = self

    def warmup(self):
        """模型预热方法"""
        print('===== [extensions] 正在对模型进行预热 =====')
        _, _ = self.model, self.class_labels
        print('===== [extensions] 模型预热完成 =====')

    @property
    def model(self):
        """延迟加载多任务模型"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    print('===== [extensions] 正在加载多任务分类模型 =====')
                    self._model = MultiTaskBERT(
                        str(Config.ft_bert_file), num_sent=2, num_asp=Config.num_labels
                    )
                    ckpt = torch.load(
                        str(Config.ft_bert_file) + '_mtl_heads.pt', map_location='cpu'
                    )
                    self._model.sent_head.load_state_dict(ckpt['sent_head'])
                    self._model.asp_head.load_state_dict(ckpt['asp_head'])
                    self._model.to(self.device)
                    self._model.eval()
                    print('===== [extensions] 多任务分类模型加载完成 =====')
        return self._model

    @property
    def class_labels(self):
        """延迟加载类别标签"""
        if self._class_labels is None:
            with self._lock:
                if self._class_labels is None:
                    with open(Config.class_file, encoding='utf-8') as file_obj:
                        self._class_labels = [line.strip() for line in file_obj if line.strip()]
        return self._class_labels

    def _deepseek_headers(self):
        """构造 DeepSeek 请求头（未配置 key 时返回 None，表示兜底不可用）"""
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            return None
        return {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

    @torch.inference_mode()
    def embed(self, texts):
        """计算评论文本的语义向量（共享 BERT 的 pooler_output，768 维）"""
        enc = self.tokenizer(
            texts, return_tensors='pt', truncation=True, padding='max_length', max_length=32
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output.cpu().numpy()

    @torch.inference_mode()
    def predict(self, raw_texts, use_fallback=True):
        """混合推理（冷热隔离）：热层 Redis 缓存命中即返回，miss 走模型 + 兜底，
        结果写回 Redis（热）并写入 FAISS 向量库（冷）。

        返回: [{'text', 'sentiment', 'sent_confidence', 'aspect', 'aspect_confidence', 'used_fallback'}, ...]
        """
        if not raw_texts:
            return []

        # 1. 热层命中检查
        results = {}
        misses = []
        for text in raw_texts:
            cached = redis_cache.get(text)
            if cached:
                results[text] = cached
            else:
                misses.append(text)

        # 2. miss 的走模型推理（串行化 GPU 访问，保护模型与显存）
        if misses:
            with self._infer_lock:
                enc = self.tokenizer(
                    misses, return_tensors='pt', truncation=True, padding='max_length', max_length=32
                )
                input_ids = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)
                sent_logits, asp_logits = self.model(input_ids, attention_mask)
                sent_conf, sent_pred = torch.softmax(sent_logits, dim=-1).max(dim=-1)
                asp_conf, asp_pred = torch.softmax(asp_logits, dim=-1).max(dim=-1)
                embeddings = self.embed(misses)  # 冷层语义向量

            headers = self._deepseek_headers() if use_fallback else None
            for i, text in enumerate(misses):
                aspect = self.class_labels[asp_pred[i].item()]
                ac = asp_conf[i].item()
                used = False
                if headers and ac < self._fallback_threshold:
                    try:
                        aspect = classify_one(text, self.class_labels, headers)
                        used = True
                    except Exception:  # noqa: BLE001
                        pass  # DeepSeek 失败则降级回本地结果

                item = {
                    'text': text,
                    'sentiment': self._sent_labels[sent_pred[i].item()],
                    'sent_confidence': round(sent_conf[i].item(), 4),
                    'aspect': aspect,
                    'aspect_confidence': round(ac, 4),
                    'used_fallback': used,
                }
                results[text] = item
                redis_cache.set(text, item)                          # 写热层
                vector_store.add(text, embeddings[i], {              # 写冷层
                    'sentiment': item['sentiment'],
                    'aspect': item['aspect'],
                })

        # 3. 按原始顺序返回
        return [results[t] for t in raw_texts]


# 创建扩展对象
text_clf_ext = TextClassifierExtension()
