"""
prediction - API 接口实现（多任务分类 + DeepSeek 低置信度兜底）

POST /api/v3/classify
    请求体: {"text": "评论内容"} 或 {"text": ["评论1", "评论2"], "use_fallback": true}
"""
from flask import Blueprint, request

from app.extensions import text_clf_ext

api_v3_bp = Blueprint('api_v3', __name__, url_prefix='/api/v3')


@api_v3_bp.route('/', methods=['GET', 'POST'])
def home():
    return {'code': 200, 'message': 'API v3 (多任务分类 + DeepSeek 兜底)'}


def _parse_texts(json_obj):
    """从请求体中解析文本列表，非法返回 (None, 错误信息)"""
    if not json_obj:
        return None, '请求数据格式错误，请提供 JSON 格式的请求数据'
    if 'text' not in json_obj:
        return None, '请求数据格式错误，请查看 API 接口文档确认数据格式'

    raw_text = json_obj['text']
    if isinstance(raw_text, str):
        raw_text = [raw_text] if raw_text.strip() else []
    elif isinstance(raw_text, list):
        raw_text = [str(t).strip() for t in raw_text if str(t).strip()]
    else:
        return None, 'text 字段必须为字符串或字符串列表'

    if not raw_text:
        return None, '请求的文本内容为空'
    return raw_text, None


@api_v3_bp.route('/classify', methods=['POST'])
def classify():
    json_obj = request.get_json(force=True, silent=True)
    raw_text, err = _parse_texts(json_obj)
    if err:
        return {'code': 400, 'message': err}, 400

    use_fallback = json_obj.get('use_fallback', True)
    results = text_clf_ext.predict(raw_text, use_fallback=use_fallback)
    return {'code': 200, 'results': results}


@api_v3_bp.route('/classify_from_file', methods=['POST'])
def classify_from_file():
    text_file = request.files.get('text_file')

    if not text_file or text_file.filename == '':
        return {'code': 400, 'message': '请求数据错误，请上传保存分类文本的文件'}, 400

    try:
        content = text_file.read().decode('utf-8', errors='ignore')
    except (UnicodeDecodeError, AttributeError, IOError):
        return {'code': 400, 'message': '文件解析失败，请确保文件为 UTF-8 编码的文本文件'}, 400

    raw_text = [line.strip() for line in content.splitlines() if line.strip()]
    if not raw_text:
        return {'code': 400, 'message': '上传的文件内容为空'}, 400

    results = text_clf_ext.predict(raw_text)
    return {'code': 200, 'results': results}
