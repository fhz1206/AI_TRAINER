from flask import Blueprint, render_template, redirect, request, jsonify, session
from blueprints.utils import login_required, get_current_username
from database import get_user_by_id

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def root():
    """根路径：直接跳转首页"""
    return redirect('/home')


@main_bp.route('/home')
def homepage():
    """公开首页，无需登录，始终展示主页"""
    if 'user_id' in session:
        # 已登录用户也可以看到首页，右上角显示导航
        return render_template('homepage.html', username=session.get('username', ''))
    return render_template('homepage.html')


@main_bp.route('/train')
@login_required
def train_page():
    """训练面板（原 index.html）"""
    user = get_user_by_id(session['user_id'])
    if user is None:
        session.clear()
        return redirect('/home')
    return render_template('index.html', username=user['username'])


@main_bp.route('/login')
def login_page():
    """登录页面"""
    if 'user_id' in session:
        return redirect('/train')
    return render_template('login.html')


@main_bp.route('/register')
def register_page():
    """注册页面（复用登录页，默认切换到注册标签）"""
    if 'user_id' in session:
        return redirect('/train')
    return render_template('login.html', default_tab='register')


@main_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    """退出登录"""
    session.clear()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': 'success', 'message': '已退出登录'})
    return redirect('/home')


@main_bp.route('/check')
@login_required
def check_page():
    """测试页面入口"""
    return render_template('check.html', username=get_current_username() or '用户')