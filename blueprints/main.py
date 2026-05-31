from flask import Blueprint, render_template, redirect, request, jsonify, session  # 补充导入session
from blueprints.utils import login_required, get_current_username
from database import get_user_by_id

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
@login_required
def index():
    """主页面"""
    user = get_user_by_id(session['user_id'])
    if user is None:
        session.clear()
        return redirect('/login')
    return render_template('index.html', username=user['username'])

@main_bp.route('/login')
def login_page():
    """登录页面"""
    if 'user_id' in session:
        return redirect('/')
    return render_template('login.html')

@main_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    """退出登录"""
    session.clear()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': 'success', 'message': '已退出登录'})
    return redirect('/login')

@main_bp.route('/check')
@login_required
def check_page():
    """测试页面入口"""
    return render_template('check.html', username=get_current_username() or '用户')