# 电商评论情感与维度分类系统

> 一个「多任务深度学习 + 大模型辅助 + 前后端分离」的电商评论智能分析系统。
> 对单条评论识别 **好评/差评** 与 **物流/质量/价格/客服/包装** 等评价维度，也能对一批评论做商品综合打分，配套 3D 渐变 Web 前端、用户注册登录与历史记录。

---

## ✨ 核心功能

- **多任务分类**：共享 BERT，同时输出情感（2 类）+ 评价维度（6 类）
- **批量商品分析**：一批评论 → 综合评分（0~5 星）+ 各维度好评率 + 差评痛点
- **大模型辅助**：DeepSeek 伪标注冷启动数据 + 低置信度样本推理兜底
- **冷热隔离存储**：Redis 热缓存 + SQLite 结构化 + FAISS 冷层语义检索
- **异步任务**：线程池 + GPU 信号量限流，防高并发
- **用户体系**：注册登录 + 历史评论记录

## 🏗 系统架构

```
数据层：京东评论 CSV + EPRSTMT 语料 → DeepSeek 伪标注（单次/定向/多票表决）
   ↓
模型层：共享 BERT encoder ─┬─ 情感头(2类)
                          └─ 维度头(6类)   ← class_weight + 早停 + 保存最优
   ↓
服务层：Flask（/classify /analyze /similar /auth/* /history /async/*）
        └─ 低置信度样本 → DeepSeek 兜底
   ↓
前端层：3D 渐变 SPA（单条分类 · 批量打分 · 登录注册 · 历史记录）
```

## 📦 技术栈

- Python 3.12 / PyTorch 2.8 / Transformers(ModelScope) / Flask
- 存储：Redis（热缓存）+ SQLite（结构化）+ FAISS（冷层向量）
- 大模型：DeepSeek（deepseek-chat）
- 前端：原生 HTML/CSS/JS（单文件、零依赖）

## 🚀 快速开始

```bash
# 1. 准备环境
conda create -n tmf python=3.12
conda activate tmf
pip install -r requirements.txt

# 2. 启动 Web 服务（默认 http://127.0.0.1:8000）
python wsgi.py

# 3. 命令行推理
python -m src.predict_mtl "物流很快第二天就到了"

# 4. 重新训练（早停 + 保存最优）
python -m src.model_train_mtl --epochs 8 --patience 2
```

## 🔌 主要接口

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| POST | `/api/v3/classify` | 单条/批量分类（含 DeepSeek 兜底） |
| POST | `/api/v3/analyze` | 批量商品维度分析 + 打分 |
| POST | `/api/v3/similar` | 语义检索相似评论（FAISS） |
| POST | `/api/auth/register` / `login` / `logout` | 注册 / 登录 / 退出 |
| GET | `/api/auth/me` | 当前登录用户 |
| POST / GET | `/api/history` | 保存 / 查询历史评论 |
| POST | `/api/async/classify` / `analyze` | 异步任务（防高并发） |
| GET | `/api/async/result/<task_id>` | 查询异步任务结果 |

## 📁 目录结构

```
tmf_v1/
├── wsgi.py                 # Web 启动入口
├── requirements.txt        # 依赖
├── 项目介绍.md              # 详细项目文档
├── src/                    # 离线模型侧
│   ├── config.py           # 配置
│   ├── model_train_mtl.py  # 多任务训练（早停+保存最优+续训）
│   ├── predict_mtl.py      # 命令行推理（含兜底）
│   ├── pseudo_label*.py    # DeepSeek 伪标注（单次/定向/多票表决）
│   └── utils.py            # 通用工具
├── app/                    # 在线服务侧（Flask）
│   ├── __init__.py         # 应用工厂
│   ├── extensions.py       # 多任务模型加载 + 混合推理
│   ├── db.py               # SQLite 访问层
│   ├── storage.py          # 冷热隔离存储（Redis + FAISS）
│   ├── auth.py / history.py  # 用户 / 历史
│   ├── analyze.py          # 批量分析 + 相似检索
│   ├── async_tasks.py      # 异步任务处理
│   └── v3/prediction.py    # 分类接口
├── front/index.html        # 前端（3D 渐变 SPA）
└── data/                   # 数据（训练/验证语料 + 类别清单）
```

## 📊 模型与数据

- **模型**：`bert-base-chinese` 多任务（共享 encoder + 双头），情感 ~90%、维度 ~72.5%
- **情感数据**：京东评论 CSV 清洗后 4.5 万条（好评/差评）
- **维度数据**：EPRSTMT 电商语料，经 DeepSeek 伪标注 + 多票表决清洗，1.16 万条（物流/质量/价格/客服/包装/其它）

## 📖 详细文档

完整的技术点、数据流程、伪标注细节见 [项目介绍.md](项目介绍.md)。
