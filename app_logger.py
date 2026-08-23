"""
app_logger.py — 全局请求日志系统（内存友好 + 磁盘有界）

相比旧版（app.py 内联实现）的改进：
1. 磁盘有界：使用 RotatingFileHandler 按大小轮转，单文件 5MB、保留 5 个备份，
   不再产生无限增长的 app.log。
2. 内存友好：不再对每个请求调用 get_data() 全量缓冲请求体/响应体，
   请求大小直接取 Content-Length，响应大小优先取 Content-Length；
   正文仅对 ≤256 字节的极小请求采样，大请求/流式响应一律跳过。
3. 安全：不再记录全部请求头（含 Cookie/Authorization 等敏感信息），
   仅记录方法、路径、类型、大小等元信息。
"""
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

from flask import request, session

# 日志目录与文件（统一放在 logs/ 下）
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
_LOG_FILE = os.path.join(_LOG_DIR, 'app.log')
_LOG_MAX_BYTES = 5 * 1024 * 1024  # 单文件上限 5MB
_LOG_BACKUP_COUNT = 5             # 保留 app.log.1 ~ app.log.5

# 请求正文采样上限：超过该大小一律不读入内存
_MAX_BODY_SAMPLE = 256


def init_app_logger(app):
    """初始化全局请求日志，需在 app 创建后调用；可重复调用（幂等）"""
    os.makedirs(_LOG_DIR, exist_ok=True)

    logger = logging.getLogger('app_access')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 幂等：重复初始化时先清空旧 handler，避免日志重复写入
    logger.handlers.clear()

    handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding='utf-8',
    )
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    app.before_request(_log_before_request)
    app.after_request(_log_after_request)
    return logger


def _log_before_request():
    """请求开始：只记录时间戳与请求大小，绝不缓冲请求体"""
    request._log_start_time = time.time()
    try:
        # content_length 直接来自请求头，无需将请求体读入内存
        request._log_req_bytes = request.content_length or 0
    except Exception:
        request._log_req_bytes = 0


def _safe_response_size(response):
    """获取响应体大小：优先用 Content-Length，避免物化整个响应体"""
    try:
        if response.content_length is not None:
            return response.content_length
        if getattr(response, 'is_streamed', False):
            return 0
        return len(response.get_data() or b'')
    except Exception:
        return 0


def _safe_request_body_sample():
    """正文采样：仅对极小请求读取，用于调试；大请求/未知大小一律跳过"""
    if request.method not in ('POST', 'PUT', 'PATCH'):
        return ''
    ct = request.content_type or ''
    length = request.content_length
    if 'multipart/form-data' in ct:
        return f'<multipart, {length or "?"} bytes>'
    if length is None or length > _MAX_BODY_SAMPLE:
        return f'<skipped, {length or "?"} bytes>'
    try:
        return request.get_data(as_text=True)[:_MAX_BODY_SAMPLE]
    except Exception:
        return '<unreadable>'


def _log_after_request(response):
    start = getattr(request, '_log_start_time', time.time())
    duration_ms = (time.time() - start) * 1000
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # ---- 带宽统计（不读取请求/响应体，大小均来自 Content-Length） ----
    try:
        uid = session.get('user_id')
        if uid:
            req_bytes = getattr(request, '_log_req_bytes', 0)
            resp_bytes = _safe_response_size(response)
            from state import track_bandwidth
            exceeded, _, limit = track_bandwidth(uid, req_bytes + resp_bytes)
            if exceeded:
                response.headers['X-RateLimit-Limit'] = str(limit)
                response.headers['X-RateLimit-Remaining'] = '0'
    except Exception:
        pass

    # 用户信息
    user_id = ''
    try:
        user_id = str(session.get('user_id', ''))
    except Exception:
        pass

    # 正文采样（仅小请求）
    body = _safe_request_body_sample()
    req_bytes = getattr(request, '_log_req_bytes', 0)
    resp_bytes = _safe_response_size(response)

    log_lines = [
        f'[REQUEST]  {now_str}',
        f'  IP:       {request.remote_addr}',
        f'  User:     {user_id or "(anonymous)"}',
        f'  Method:   {request.method}',
        f'  Path:     {request.full_path}',
        f'  Type:     {request.content_type or "-"}',
        f'  Size:     {req_bytes} bytes in / {resp_bytes} bytes out',
        f'  Body:     {body}',
        f'[RESPONSE] {response.status_code} ({duration_ms:.1f}ms)',
        f'{"="*80}',
    ]
    logging.getLogger('app_access').info('\n'.join(log_lines))
    return response
