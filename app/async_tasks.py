"""
async_tasks - 异步任务处理（防止高并发压垮推理服务）

策略:
    1. 请求入队后立刻返回 task_id，Web 请求不再阻塞在模型推理上；
    2. 后台线程池受限并发执行，信号量串行化 GPU 推理（保护显存/模型）；
    3. 队列长度受限，超限返回 429，防止任务无限堆积；
    4. 客户端轮询 GET /api/async/result/<task_id> 获取结果。
"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, request

from app.analyze import aggregate
from app.extensions import text_clf_ext

MAX_WORKERS = 2        # 工作线程数
MAX_QUEUE = 50         # 队列上限（防堆积）
GPU_SLOTS = 1          # 同时跑 GPU 推理的任务数（串行化，保护显存）

_gpu_semaphore = threading.Semaphore(GPU_SLOTS)
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
_tasks = {}
_lock = threading.Lock()


def _run(task_id, kind, payload):
    with _lock:
        _tasks[task_id]['status'] = 'running'
    try:
        with _gpu_semaphore:  # 串行化 GPU 推理
            if kind == 'classify':
                result = text_clf_ext.predict(
                    payload['texts'], use_fallback=payload.get('use_fallback', True)
                )
            elif kind == 'analyze':
                result = aggregate(text_clf_ext.predict(payload['texts']))
            else:
                result = None
        with _lock:
            _tasks[task_id].update({'status': 'done', 'result': result, 'finished_at': time.time()})
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _tasks[task_id].update({'status': 'error', 'result': str(exc)})


def submit(kind, payload):
    """提交异步任务；队列满返回 None"""
    with _lock:
        pending = sum(1 for t in _tasks.values() if t['status'] in ('pending', 'running'))
        if pending >= MAX_QUEUE:
            return None
        task_id = uuid.uuid4().hex
        _tasks[task_id] = {'status': 'pending', 'result': None, 'created_at': time.time()}
    _executor.submit(_run, task_id, kind, payload)
    return task_id


def get(task_id):
    with _lock:
        t = _tasks.get(task_id)
        if not t:
            return {'status': 'not_found'}
        return {'status': t['status'], 'result': t['result']}


def stats():
    with _lock:
        by_status = {}
        for t in _tasks.values():
            by_status[t['status']] = by_status.get(t['status'], 0) + 1
        return {
            'total': len(_tasks),
            'by_status': by_status,
            'max_queue': MAX_QUEUE,
            'max_workers': MAX_WORKERS,
        }


# ---------- 路由 ----------
async_bp = Blueprint('async', __name__, url_prefix='/api/async')


def _parse_texts(json_obj):
    texts = json_obj.get('text') if 'text' in json_obj else json_obj.get('texts')
    if isinstance(texts, str):
        texts = texts.splitlines()
    return [str(t).strip() for t in (texts or []) if str(t).strip()]


@async_bp.route('/classify', methods=['POST'])
def async_classify():
    json_obj = request.get_json(silent=True) or {}
    texts = _parse_texts(json_obj)
    if not texts:
        return {'code': 400, 'message': '请提供评论文本'}, 400
    task_id = submit('classify', {'texts': texts, 'use_fallback': json_obj.get('use_fallback', True)})
    if task_id is None:
        return {'code': 429, 'message': '任务队列已满，请稍后重试'}, 429
    return {'code': 200, 'task_id': task_id}


@async_bp.route('/analyze', methods=['POST'])
def async_analyze():
    json_obj = request.get_json(silent=True) or {}
    texts = _parse_texts(json_obj)
    if not texts:
        return {'code': 400, 'message': '请提供评论列表'}, 400
    task_id = submit('analyze', {'texts': texts})
    if task_id is None:
        return {'code': 429, 'message': '任务队列已满，请稍后重试'}, 429
    return {'code': 200, 'task_id': task_id}


@async_bp.route('/result/<task_id>', methods=['GET'])
def async_result(task_id):
    return {'code': 200, **get(task_id)}


@async_bp.route('/stats', methods=['GET'])
def async_stats():
    return {'code': 200, **stats()}
