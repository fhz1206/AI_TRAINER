from flask import Blueprint, render_template, redirect, request, jsonify, session  # 补充导入session
from blueprints.utils import login_required, get_current_username
from database import get_user_by_id

study_bp = Blueprint('study', __name__)

@study_bp.route('/study')
@login_required
def index():
    """主页面"""
    user = get_user_by_id(session['user_id'])
    if user is None:
        session.clear()
        return redirect('/login')
    return render_template('ai.html', username=user['username'])