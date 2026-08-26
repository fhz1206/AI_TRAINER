// profile.js —— 由模板 profile.html 的内联脚本迁出（保持原执行顺序）
        // 从父页面获取用户名（如果是在iframe中）或直接使用
        const currentUsername = window.__PAGE__.username;

        // 加载用户信息
        async function loadProfile() {
            try {
                const res = await fetch('/api/profile/info');
                const data = await res.json();
                if (data.status === 'success') {
                    const u = data.user;
                    document.getElementById('displayName').textContent = u.username;
                    document.getElementById('displayMeta').textContent =
                        `注册于 ${u.created_at?.split('T')[0] || '未知'} · 上次登录 ${u.last_login?.split('T')[0] || '首次'}`;
                    document.querySelectorAll('.stat-card')[0].querySelector('.num').textContent = u.file_count;
                    document.querySelectorAll('.stat-card')[1].querySelector('.num').textContent = u.model_count;
                }
            } catch (e) {
                document.getElementById('displayMeta').textContent = '加载失败';
            }
        }

        // 加载行为日志
        async function loadActivityLogs() {
            const list = document.getElementById('logList');
            try {
                const res = await fetch('/api/profile/activity_logs');
                const data = await res.json();
                if (data.status !== 'success' || !data.logs || data.logs.length === 0) {
                    list.innerHTML = '<li class="log-empty">暂无行为记录</li>';
                    return;
                }
                const iconMap = {
                    'upload': '📤', 'train': '🧠', 'test': '🧪',
                    'profile': '⚙️', 'delete': '🗑️', 'download': '⬇️'
                };
                list.innerHTML = data.logs.map(log => {
                    const icon = iconMap[log.activity_type] || '📌';
                    const detail = log.detail ? `<br><span style="font-size:12px;color:var(--text-muted)">${log.detail}</span>` : '';
                    return `<li class="log-item">
                        <div class="log-icon">${icon}</div>
                        <div class="log-content">
                            <div class="log-desc">${log.description}${detail}</div>
                            <div class="log-time">${log.created_at}</div>
                        </div>
                    </li>`;
                }).join('');

                // 统计测试记录数
                const testCount = data.logs.filter(l => l.activity_type === 'test').length;
                document.querySelectorAll('.stat-card')[2].querySelector('.num').textContent = testCount;
            } catch (e) {
                list.innerHTML = '<li class="log-empty">❌ 加载失败</li>';
            }
        }

        // 修改用户名
        async function updateUsername() {
            const input = document.getElementById('newUsername');
            const btn = event.target;
            const msg = document.getElementById('usernameMsg');
            const name = input.value.trim();
            if (!name || name.length < 2 || name.length > 20) {
                msg.className = 'msg error'; msg.textContent = '用户名长度应为2-20个字符'; return;
            }
            btn.disabled = true; btn.textContent = '⏳ 保存中...';
            try {
                const res = await fetch('/api/profile/update_username', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: name })
                });
                const data = await res.json();
                msg.className = 'msg ' + (data.status === 'success' ? 'success' : 'error');
                msg.textContent = data.message;
                if (data.status === 'success') {
                    document.getElementById('displayName').textContent = name;
                    document.getElementById('headerUsername').textContent = name;
                }
            } catch (e) {
                msg.className = 'msg error'; msg.textContent = '请求失败: ' + e.message;
            }
            btn.disabled = false; btn.textContent = '保存修改';
        }

        // 修改密码
        async function updatePassword() {
            const oldPw = document.getElementById('oldPassword').value;
            const newPw = document.getElementById('newPassword').value;
            const confirmPw = document.getElementById('confirmPassword').value;
            const msg = document.getElementById('passwordMsg');
            if (!oldPw || !newPw) {
                msg.className = 'msg error'; msg.textContent = '请填写所有密码字段'; return;
            }
            if (newPw.length < 4) {
                msg.className = 'msg error'; msg.textContent = '新密码至少4个字符'; return;
            }
            if (newPw !== confirmPw) {
                msg.className = 'msg error'; msg.textContent = '两次密码不一致'; return;
            }
            const btn = event.target;
            btn.disabled = true; btn.textContent = '⏳ 修改中...';
            try {
                const res = await fetch('/api/profile/update_password', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ old_password: oldPw, new_password: newPw })
                });
                const data = await res.json();
                msg.className = 'msg ' + (data.status === 'success' ? 'success' : 'error');
                msg.textContent = data.message;
                if (data.status === 'success') {
                    document.getElementById('oldPassword').value = '';
                    document.getElementById('newPassword').value = '';
                    document.getElementById('confirmPassword').value = '';
                }
            } catch (e) {
                msg.className = 'msg error'; msg.textContent = '请求失败: ' + e.message;
            }
            btn.disabled = false; btn.textContent = '修改密码';
        }

        // 切换选项卡
        function switchTab(tab) {
            document.querySelectorAll('.tab-nav button').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab' + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add('active');
        }

        // 初始化
        loadProfile();
        loadActivityLogs();
