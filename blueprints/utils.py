from flask import session, redirect, url_for
from config import Config

def login_required(f):
    """登录验证装饰器，和原有逻辑完全一致"""
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

def get_current_user_id():
    """获取当前登录用户ID"""
    return session.get('user_id')

def get_current_username():
    """获取当前登录用户名"""
    return session.get('username')