"""
auth - 用户注册 / 登录 / 退出

使用 Flask session 维持登录态（依赖 app.config 的 SECRET_KEY）。
"""
from flask import Blueprint, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from app.db import create_user, get_user

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if len(username) < 2 or len(password) < 6:
        return {'code': 400, 'message': '用户名至少 2 位，密码至少 6 位'}, 400

    if create_user(username, generate_password_hash(password)):
        return {'code': 200, 'message': '注册成功，请登录'}
    return {'code': 400, 'message': '用户名已存在'}, 400


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    user = get_user(username)
    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return {'code': 200, 'message': '登录成功', 'user': {'id': user['id'], 'username': user['username']}}
    return {'code': 401, 'message': '用户名或密码错误'}, 401


@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return {'code': 200, 'message': '已退出登录'}


@auth_bp.route('/me', methods=['GET'])
def me():
    if 'user_id' in session:
        return {'code': 200, 'user': {'id': session['user_id'], 'username': session['username']}}
    return {'code': 200, 'user': None}
