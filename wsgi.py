"""
wsgi - Web 项目启动入口

Author: 骆昊
Version: 0.0.1
"""
from pathlib import Path

from flask import send_file

from app import create_app

app = create_app()

FRONT_DIR = Path(__file__).resolve().parent / 'front'


@app.route('/')
def index():
    """首页"""
    return send_file(FRONT_DIR / 'index.html')


if __name__ == '__main__':
    # 注意: 下面运行服务的方式仅供测试使用! 生产环境请使用 gunicorn / waitress
    # 工业级部署绝对不要把 Gunicorn 或 Waitress 直接暴露给外网, 前面要有 Nginx
    # gunicorn -w 4 -b 0.0.0.0:8000 --preload wsgi:app
    # 如果模型加载需要较长的时间, 可以使用延迟加载或 -t 参数来增加启动时间
    app.run(host='127.0.0.1', port=8000)
