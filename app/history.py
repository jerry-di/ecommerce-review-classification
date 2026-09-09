"""
history - 历史评论记录（需登录）
"""
from flask import Blueprint, request, session

from app.db import add_history, get_history

history_bp = Blueprint('history', __name__, url_prefix='/api/history')


def _current_user_id():
    return session.get('user_id')


@history_bp.route('', methods=['POST'])
def save():
    user_id = _current_user_id()
    if user_id is None:
        return {'code': 401, 'message': '请先登录'}, 401

    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return {'code': 400, 'message': '评论内容为空'}, 400

    add_history(
        user_id,
        text,
        data.get('sentiment'),
        data.get('aspect'),
        data.get('sentiment_conf'),
        data.get('aspect_conf'),
        data.get('used_fallback'),
    )
    return {'code': 200, 'message': '已保存'}


@history_bp.route('', methods=['GET'])
def list_history():
    user_id = _current_user_id()
    if user_id is None:
        return {'code': 401, 'message': '请先登录'}, 401

    rows = get_history(user_id)
    return {'code': 200, 'history': [dict(r) for r in rows]}
