"""
config - 训练模型相关配置类
"""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """配置类"""
    pretrained_model:  str = 'google-bert/bert-base-chinese'
    num_labels:        int = 6

    class_file:        Path = BASE_DIR / 'data' / 'eprstmt_class.txt'

    # 评价维度六分类数据（DeepSeek 伪标注，多票表决清洗后）
    train_raw_file:    Path = BASE_DIR / 'data/raw' / 'eprstmt_aspect_train.txt'
    valid_raw_file:    Path = BASE_DIR / 'data/raw' / 'eprstmt_aspect_valid.txt'
    unlabeled_file:    Path = BASE_DIR / 'data/raw' / 'eprstmt_unlabeled.txt'
    pseudo_labeled_file: Path = BASE_DIR / 'data/raw' / 'eprstmt_aspect_labeled.txt'

    # 情感二分类数据（京东评论 CSV 清洗后）
    sentiment_train_file: Path = BASE_DIR / 'data/raw' / 'jd_sentiment_train.txt'
    sentiment_valid_file: Path = BASE_DIR / 'data/raw' / 'jd_sentiment_valid.txt'

    # 训练产物
    ft_bert_file:      Path = BASE_DIR / 'model' / 'ft_bert_model'
