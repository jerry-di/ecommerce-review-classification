"""
storage - 冷热隔离存储层

    热层 Redis  : 缓存分类结果（内存、快、TTL 自动淘汰）
    温层 SQLite : 用户账号 + 历史记录（见 app/db.py）
    冷层 FAISS  : 全量评论的 768 维语义向量（持久化、用于相似评论检索）

设计思路:
    - 高频热数据（最近分类过的评论）放 Redis，命中即秒回，不重复算；
    - 全量冷数据（所有评论的语义向量）放 FAISS 落盘，按需做语义检索。
"""
import json
import threading
from pathlib import Path

import faiss
import numpy as np
import redis

BASE_DIR = Path(__file__).resolve().parent.parent
DIM = 768
PERSIST_DIR = BASE_DIR / 'model' / 'review_vectors'


# ---------- 热层：Redis 缓存 ----------
class RedisCache:
    """Redis 热缓存；Redis 不可用时自动降级为不可用（不影响主流程）"""

    def __init__(self, host='localhost', port=6379, db=0):
        self._client = None
        self._host, self._port, self._db = host, port, db

    @property
    def client(self):
        if self._client is None:
            try:
                # protocol=2 兼容旧版 Redis（5.x 不支持 RESP3 的 HELLO）
                c = redis.Redis(host=self._host, port=self._port, db=self._db,
                                decode_responses=True, protocol=2)
                c.ping()
                self._client = c
            except Exception:
                self._client = None
        return self._client

    @property
    def available(self):
        return self.client is not None

    def get(self, text):
        """查缓存，未命中返回 None"""
        try:
            raw = self.client.get(f'clf:{text}')
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, text, result, ttl=3600):
        """写缓存，TTL 默认 1 小时"""
        try:
            self.client.set(f'clf:{text}', json.dumps(result, ensure_ascii=False), ex=ttl)
        except Exception:
            pass


# ---------- 冷层：FAISS 向量库 ----------
class VectorStore:
    """语义向量冷存储：存评论的 BERT 向量，支持相似评论检索，落盘持久化"""

    def __init__(self, dim=DIM, persist_dir=PERSIST_DIR):
        self.dim = dim
        self.persist_dir = persist_dir
        self.index = faiss.IndexFlatIP(dim)  # 内积（向量已 L2 归一化，等价余弦相似度）
        self.texts = []   # 与 index 行对应的评论文本
        self.metas = []   # 与 index 行对应的元数据
        self._lock = threading.Lock()
        self._load()

    def add(self, text, embedding, meta=None):
        with self._lock:  # FAISS 非线程安全，串行化写入
            emb = np.asarray(embedding, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(emb)
            self.index.add(emb)
            self.texts.append(text)
            self.metas.append(meta or {})
            self._save()

    def search(self, embedding, k=5):
        """按语义找最相似的 k 条评论"""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            emb = np.asarray(embedding, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(emb)
            dists, idxs = self.index.search(emb, min(k, self.index.ntotal))
            out = []
            for j, i in enumerate(idxs[0]):
                if i < 0 or i >= len(self.texts):
                    continue
                out.append({
                    'text': self.texts[i],
                    'meta': self.metas[i],
                    'similarity': round(float(dists[0][j]), 4),
                })
            return out

    def count(self):
        return int(self.index.ntotal)

    # ---- 持久化 ----
    def _embeddings(self):
        if self.index.ntotal == 0:
            return np.empty((0, self.dim), dtype='float32')
        # 从 FlatIP 索引取回原始向量（重建所需）
        return np.asarray(self.index.reconstruct_n(0, self.index.ntotal), dtype='float32')

    def _save(self):
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.persist_dir / 'embeddings.npy', self._embeddings())
            with open(self.persist_dir / 'records.json', 'w', encoding='utf-8') as f:
                json.dump([{'text': t, 'meta': m} for t, m in zip(self.texts, self.metas)],
                          f, ensure_ascii=False)
        except Exception:
            pass

    def _load(self):
        emb_path = self.persist_dir / 'embeddings.npy'
        rec_path = self.persist_dir / 'records.json'
        if not (emb_path.exists() and rec_path.exists()):
            return
        try:
            embs = np.load(emb_path)
            with open(rec_path, encoding='utf-8') as f:
                records = json.load(f)
            for emb, rec in zip(embs, records):
                e = np.asarray(emb, dtype='float32').reshape(1, -1)
                faiss.normalize_L2(e)
                self.index.add(e)
                self.texts.append(rec['text'])
                self.metas.append(rec.get('meta', {}))
        except Exception:
            self.index = faiss.IndexFlatIP(self.dim)
            self.texts, self.metas = [], []


# 全局单例
redis_cache = RedisCache()
vector_store = VectorStore()
