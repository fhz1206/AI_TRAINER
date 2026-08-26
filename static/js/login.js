// login.js —— 由模板 login.html 的内联脚本迁出（保持原执行顺序）
        const defaultTab = window.__PAGE__.defaultTab || 'login';
        function switchTab(tab) {
            document.getElementById('tabLogin').classList.toggle('active', tab === 'login');
            document.getElementById('tabRegister').classList.toggle('active', tab === 'register');
            document.getElementById('loginForm').style.display = tab === 'login' ? 'block' : 'none';
            document.getElementById('registerForm').style.display = tab === 'register' ? 'block' : 'none';
            document.getElementById('message').style.display = 'none';
        }
        // 初始化：根据URL参数切换到对应标签
        document.addEventListener('DOMContentLoaded', () => switchTab(defaultTab));

        function showMessage(text, type) {
            const msg = document.getElementById('message');
            msg.textContent = text;
            msg.className = 'message ' + type;
            msg.style.display = 'block';
        }

        async function handleLogin(e) {
            e.preventDefault();
            const btn = document.getElementById('loginBtn');
            btn.disabled = true;
            btn.textContent = '⏳ 登录中...';
            
            const username = document.getElementById('loginUsername').value.trim();
            const password = document.getElementById('loginPassword').value;
            
            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                
                if (data.status === 'success') {
                    showMessage('✅ 登录成功，正在跳转...', 'success');
                    setTimeout(() => { window.location.href = '/train'; }, 800);
                } else {
                    showMessage('❌ ' + data.message, 'error');
                    btn.disabled = false;
                    btn.textContent = '🚀 登录';
                }
            } catch (err) {
                showMessage('❌ 登录失败: ' + err.message, 'error');
                btn.disabled = false;
                btn.textContent = '🚀 登录';
            }
        }

        async function handleRegister(e) {
            e.preventDefault();
            const btn = document.getElementById('registerBtn');
            btn.disabled = true;
            btn.textContent = '⏳ 注册中...';

            const username = document.getElementById('regUsername').value.trim();
            const password = document.getElementById('regPassword').value;
            const confirm = document.getElementById('regConfirm').value;

            if (password !== confirm) {
                showMessage('❌ 两次密码不一致', 'error');
                btn.disabled = false;
                btn.textContent = '📝 注册';
                return;
            }
            
            try {
                const res = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                
                if (data.status === 'success') {
                    showMessage('✅ 注册成功！请登录', 'success');
                    setTimeout(() => { switchTab('login'); }, 1000);
                } else {
                    showMessage('❌ ' + data.message, 'error');
                }
            } catch (err) {
                showMessage('❌ 注册失败: ' + err.message, 'error');
            }
            
            btn.disabled = false;
            btn.textContent = '📝 注册';
        }

        // 按 Enter 自动提交
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                if (document.getElementById('loginForm').style.display !== 'none') {
                    document.getElementById('loginForm').requestSubmit();
                } else {
                    document.getElementById('registerForm').requestSubmit();
                }
            }
        });
