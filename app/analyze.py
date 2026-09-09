"""
analyze - 批量评论的商品维度分析 + 打分

POST /api/v3/analyze
    请求体: {"texts": ["评论1", "评论2", ...]}
    返回: 综合好评率/评分 + 各维度（物流/质量/价格/客服/包装）的好评率、提及数、正负样例
"""
from flask import Blueprint, request

from app.extensions import text_clf_ext
from app.storage import vector_store

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/v3')

ASPECTS = ['物流', '质量', '价格', '客服', '包装', '其它']


def aggregate(results):
    """把逐条分类结果聚合成商品维度分析报告"""
    stats = {a: {'pos': [], 'neg': []} for a in ASPECTS}
    total_pos = 0

    for r in results:
        aspect = r['aspect'] if r['aspect'] in stats else '其它'
        if r['sentiment'] == '好评':
            stats[aspect]['pos'].append(r['text'])
            total_pos += 1
        else:
            stats[aspect]['neg'].append(r['text'])

    total = len(results)
    dimensions = []
    for aspect in ASPECTS:
        pos, neg = stats[aspect]['pos'], stats[aspect]['neg']
        mentions = len(pos) + len(neg)
        if mentions == 0:
            continue
        rate = len(pos) / mentions
        dimensions.append({
            'aspect': aspect,
            'mentions': mentions,
            'positive': len(pos),
            'negative': len(neg),
            'positive_rate': round(rate, 4),
            'score': round(rate * 5, 1),
            'sample_positive': pos[:2],
            'sample_negative': neg[:2],
        })

    overall_rate = total_pos / total if total else 0.0
    return {
        'total': total,
        'positive': total_pos,
        'negative': total - total_pos,
        'overall_positive_rate': round(overall_rate, 4),
        'overall_score': round(overall_rate * 5, 1),
        'dimensions': dimensions,
    }


@analysis_bp.route('/analyze', methods=['POST'])
def analyze():
    json_obj = request.get_json(silent=True) or {}
    texts = json_obj.get('texts')

    if isinstance(texts, str):
        texts = texts.splitlines()
    texts = [str(t).strip() for t in (texts or []) if str(t).strip()]
    if not texts:
        return {'code': 400, 'message': '请提供评论列表'}, 400

    results = text_clf_ext.predict(texts)
    report = aggregate(results)
    report['code'] = 200
    return report


@analysis_bp.route('/similar', methods=['POST'])
def similar():
    """语义检索：输入一条评论，从冷层 FAISS 向量库找语义最相似的评论"""
    json_obj = request.get_json(silent=True) or {}
    text = (json_obj.get('text') or '').strip()
    if not text:
        return {'code': 400, 'message': '请提供评论文本'}, 400

    k = int(json_obj.get('k', 5))
    embedding = text_clf_ext.embed([text])[0]
    hits = vector_store.search(embedding, k)
    return {
        'code': 200,
        'query': text,
        'indexed': vector_store.count(),
        'results': hits,
    }
