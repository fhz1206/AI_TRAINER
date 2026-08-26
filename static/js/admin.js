// admin.js —— 由模板 admin.html 的内联脚本迁出（保持原执行顺序）
        let selectedUserId = null;
        let allUsers = [];
        let allLogs = [];

        function switchTab(tab) {
            document.querySelectorAll('.admin-tabs button').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab' + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add('active');
            // 切换到日志标签时重新获取数据
            if (tab === 'logs') loadLogs();
            if (tab === 'device') loadDeviceStatus();
            if (tab === 'queue') loadQueue();
            if (tab === 'bandwidth') loadBandwidth();
        }

        // 加载用户列表
        async function loadUsers() {
            try {
                const res = await fetch('/admin/api/users');
                const data = await res.json();
                if (data.status !== 'success') return;
                allUsers = data.users;
                renderUsers(allUsers);
                updateStats();
            } catch (e) {
                document.getElementById('userTableBody').innerHTML = '<tr><td colspan="6" class="empty-state">❌ 加载失败</td></tr>';
            }
        }

        function renderUsers(users) {
            const tbody = document.getElementById('userTableBody');
            if (!users || users.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state">暂无用户</td></tr>';
                return;
            }
            tbody.innerHTML = users.map(u => {
                const roleBadge = u.role === 'admin' ? '<span class="badge-role badge-admin">管理员</span>' : '<span class="badge-role badge-user">用户</span>';
                const groupHtml = u.group_name ? `<span class="badge-group">${u.group_name}</span>` : '<span style="color:var(--text-muted)">—</span>';
                const isSelf = u.id === parseInt(String(window.__PAGE__.userId));
                const isAdminUser = u.username === 'admin';
                const canDelete = !isAdminUser && !isSelf;
                return `<tr>
                    <td>${u.id}</td>
                    <td><strong>${u.username}</strong></td>
                    <td>
                        <select class="select-group" onchange="changeRole(${u.id}, this.value)" ${isAdminUser ? 'disabled' : ''}>
                            <option value="user" ${u.role === 'user' ? 'selected' : ''}>用户</option>
                            <option value="admin" ${u.role === 'admin' ? 'selected' : ''}>管理员</option>
                        </select>
                    </td>
                    <td>
                        <div style="display:flex;gap:4px;align-items:center;">
                            <input type="text" class="select-group" style="width:100px" value="${u.group_name || ''}" 
                                   placeholder="分组名" id="groupInput_${u.id}">
                            <button class="btn-sm" onclick="saveGroup(${u.id})" ${isAdminUser ? 'disabled' : ''}>💾 保存</button>
                        </div>
                    </td>
                    <td style="color:var(--text-muted);font-size:13px;">${u.created_at || '—'}</td>
                    <td>
                        <button class="btn-sm" onclick="openResetPwd(${u.id}, '${u.username}')" ${isAdminUser ? 'disabled' : ''}>🔑 改密码</button>
                        <button class="btn-sm btn-danger" onclick="deleteUser(${u.id}, '${u.username}')" ${canDelete ? '' : 'disabled'}>🗑️ 删除</button>
                    </td>
                </tr>`;
            }).join('');
        }

        function updateStats() {
            const total = allUsers.length;
            const admins = allUsers.filter(u => u.role === 'admin').length;
            const users = total - admins;
            document.querySelectorAll('.stat-card')[0].querySelector('.num').textContent = total;
            document.querySelectorAll('.stat-card')[1].querySelector('.num').textContent = admins;
            document.querySelectorAll('.stat-card')[2].querySelector('.num').textContent = users;
        }

        function searchUsers() {
            const q = document.getElementById('searchInput').value.toLowerCase().trim();
            if (!q) { renderUsers(allUsers); return; }
            const filtered = allUsers.filter(u => 
                u.username.toLowerCase().includes(q) || (u.group_name || '').toLowerCase().includes(q)
            );
            renderUsers(filtered);
        }

        // Toast 提示消息
        function showToast(message, type) {
            const container = document.getElementById('toast');
            const el = document.createElement('div');
            el.style.cssText = [
                'padding:12px 20px;border-radius:10px;font-size:14px;font-weight:500;',
                'backdrop-filter:blur(10px);box-shadow:0 8px 30px rgba(0,0,0,0.4);',
                'animation:slideIn 0.3s ease, fadeOut 0.3s ease 2.7s forwards;',
                'border:1px solid rgba(255,255,255,0.1);',
                'pointer-events:auto;max-width:360px;'
            ].join('');
            if (type === 'success') {
                el.style.background = 'rgba(34,197,94,0.15)';
                el.style.color = '#4ade80';
                el.style.borderColor = 'rgba(34,197,94,0.3)';
            } else if (type === 'error') {
                el.style.background = 'rgba(239,68,68,0.15)';
                el.style.color = '#fca5a5';
                el.style.borderColor = 'rgba(239,68,68,0.3)';
            } else {
                el.style.background = 'rgba(99,102,241,0.15)';
                el.style.color = '#818cf8';
                el.style.borderColor = 'rgba(99,102,241,0.3)';
            }
            el.textContent = message;
            container.appendChild(el);
            setTimeout(() => { if (el.parentNode) el.parentNode.removeChild(el); }, 3000);
        }
        // 注入 toast 动画
        const ts = document.createElement('style');
        ts.textContent = '@keyframes slideIn{from{transform:translateX(100px);opacity:0}to{transform:translateX(0);opacity:1}}@keyframes fadeOut{from{opacity:1}to{opacity:0}}';
        document.head.appendChild(ts);

        async function changeRole(userId, role) {
            try {
                const res = await fetch(`/admin/api/user/${userId}/role`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ role })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    loadUsers();
                    showToast('✅ 角色已更新', 'success');
                } else {
                    showToast('❌ ' + data.message, 'error');
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        async function saveGroup(userId) {
            const input = document.getElementById('groupInput_' + userId);
            const group = input ? input.value.trim() : '';
            try {
                const res = await fetch(`/admin/api/user/${userId}/group`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ group })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    loadUsers();
                    showToast('✅ 分组已保存', 'success');
                } else {
                    showToast('❌ ' + data.message, 'error');
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        async function deleteUser(userId, username) {
            if (!confirm(`确定要删除用户「${username}」吗？此操作不可撤销！`)) return;
            try {
                const res = await fetch(`/admin/api/user/${userId}/delete`, { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    loadUsers();
                    showToast('🗑️ 用户已删除', 'success');
                } else {
                    showToast('❌ ' + data.message, 'error');
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        function openResetPwd(userId, username) {
            selectedUserId = userId;
            document.getElementById('resetPwdUser').textContent = `重置用户「${username}」的密码`;
            document.getElementById('newPwdInput').value = '';
            document.getElementById('resetPwdModal').classList.add('show');
        }

        function closeModal() {
            document.getElementById('resetPwdModal').classList.remove('show');
            selectedUserId = null;
        }

        async function confirmResetPwd() {
            const pwd = document.getElementById('newPwdInput').value;
            if (!pwd || pwd.length < 4) { showToast('⚠️ 密码至少4位', 'error'); return; }
            try {
                const res = await fetch(`/admin/api/user/${selectedUserId}/reset_password`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password: pwd })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast('🔑 密码已重置', 'success');
                    closeModal();
                } else {
                    showToast('❌ ' + data.message, 'error');
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        // 加载日志
        let logsPage = 1;
        let logsTotalPages = 1;       // 最近一次加载得到的总页数（翻页边界用）
        const LOGS_PAGE_SIZE = 100;   // 每页固定条数，接口侧另有 500 上限保护

        async function loadLogs(page) {
            // 分页拉取：每次只取一页（LIMIT/OFFSET），避免大结果撑爆内存
            try {
                if (typeof page === 'number') logsPage = page;
                const res = await fetch(`/admin/api/logs?page=${logsPage}&page_size=${LOGS_PAGE_SIZE}`);
                const data = await res.json();
                if (data.status !== 'success') return;
                allLogs = data.logs;
                logsTotalPages = Math.max(1, Math.ceil(data.total / data.page_size));
                document.getElementById('logPageInfo').textContent =
                    `第 ${data.page} / ${logsTotalPages} 页 · 共 ${data.total} 条`;
                // 翻页按钮按可用范围禁用（单页时两键均灰）
                const prevBtn = document.getElementById('logPrevBtn');
                const nextBtn = document.getElementById('logNextBtn');
                if (prevBtn) prevBtn.disabled = data.page <= 1;
                if (nextBtn) nextBtn.disabled = data.page >= logsTotalPages;
                renderLogs(allLogs);
            } catch (e) {
                document.getElementById('logList').innerHTML = '<div class="empty-state">❌ 加载失败</div>';
            }
        }

        function changeLogsPage(delta) {
            const next = logsPage + delta;
            // 双向边界：小于第一页或超出总页数一律忽略（修复单页时仍能翻到第 2 页）
            if (next < 1 || next > logsTotalPages) return;
            loadLogs(next);
        }

        // ---- 日志存储上限（-1=无上限）----
        async function loadLogLimit() {
            try {
                const res = await fetch('/admin/api/log_limits');
                const data = await res.json();
                if (data.status !== 'success') return;
                document.getElementById('logLimitInput').value = data.max_logs;
                document.getElementById('logLimitInfo').textContent =
                    data.max_logs === -1 ? '当前：无上限' : `当前：保留最新 ${data.max_logs} 条`;
            } catch (e) { console.error('load log limit failed:', e); }
        }

        async function setLogLimit() {
            const v = parseInt(document.getElementById('logLimitInput').value, 10);
            if (isNaN(v) || (v !== -1 && v < 0)) {
                alert('请输入 -1（无上限）或非负整数');
                return;
            }
            try {
                const res = await fetch('/admin/api/log_limits', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_logs: v })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    alert(v === -1 ? '已设为无上限' : `已设为保留最新 ${v} 条`);
                    loadLogLimit();
                    loadLogs(1);
                } else {
                    alert(data.message || '保存失败');
                }
            } catch (e) { alert('保存请求失败'); }
        }

        function renderLogs(logs) {
            const container = document.getElementById('logList');
            if (!logs || logs.length === 0) {
                container.innerHTML = '<div class="empty-state">暂无日志</div>';
                return;
            }
            const typeMap = { 'upload': 'upload', 'train': 'train', 'test': 'test', 'admin': 'admin', 'profile': 'profile' };
            container.innerHTML = logs.map(log => {
                const typeClass = 'log-type-' + (typeMap[log.activity_type] || 'profile');
                return `<div class="log-item">
                    <span class="log-time">${log.created_at}</span>
                    <span class="log-user">${log.username}</span>
                    <span class="log-type ${typeClass}">${log.activity_type}</span>
                    <span class="log-desc">${log.description}</span>
                </div>`;
            }).join('');
        }

        function filterLogs() {
            const q = document.getElementById('logSearchInput').value.toLowerCase().trim();
            if (!q) { renderLogs(allLogs); return; }
            const filtered = allLogs.filter(l => 
                l.description.toLowerCase().includes(q) || 
                l.username.toLowerCase().includes(q) ||
                l.activity_type.toLowerCase().includes(q)
            );
            renderLogs(filtered);
        }

        // 初始化
        loadUsers();
        loadLogs();
        loadDeviceStatus();
        loadCleanupStatus();
        loadResourceLimits();
        loadLogLimit();

        // 资源占用上限
        async function loadResourceLimits() {
            try {
                const res = await fetch('/admin/api/resource_limits');
                const data = await res.json();
                if (data.status !== 'success') return;
                const L = data.limits;
                document.getElementById('rlCpuInput').value = L.max_cpu_threads;
                document.getElementById('rlMemInput').value = L.max_memory_mb;
                const u = L.usage || {};
                const src = L.is_default ? '默认(系统一半)' : '自定义';
                document.getElementById('resourceLimitsInfo').textContent =
                    `当前配置: ${src} | 系统: ${L.system_cpu_threads} 线程 / ${L.system_memory_mb ?? '?'} MB 内存` +
                    ` | 本进程占用: ${u.memory_mb ?? '?'} MB / CPU ${u.cpu_percent ?? '-'}%` +
                    ` | 超限后新训练将被拒绝`;
            } catch (e) { console.error('resource limits failed:', e); }
        }

        async function saveResourceLimits() {
            const cpu = parseInt(document.getElementById('rlCpuInput').value || '0', 10);
            const mem = parseInt(document.getElementById('rlMemInput').value || '0', 10);
            try {
                const res = await fetch('/admin/api/resource_limits', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_cpu_threads: cpu, max_memory_mb: mem })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    alert(`资源上限已保存：CPU ${cpu} 线程 / 内存 ${mem} MB（0=不限）`);
                    loadResourceLimits();
                } else {
                    alert(data.message || '保存失败');
                }
            } catch (e) { alert('保存请求失败'); }
        }

        // 存储清理
        function fmtSize(bytes) {
            if (!bytes) return '0 B';
            const units = ['B', 'KB', 'MB', 'GB'];
            let i = 0, v = bytes;
            while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
            return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + ' ' + units[i];
        }

        async function loadCleanupStatus() {
            try {
                const res = await fetch('/admin/api/cleanup/status');
                const data = await res.json();
                if (data.status !== 'success') return;
                const c = data.cleanup;
                document.getElementById('retentionDaysInput').value = c.config.retention_days;
                document.getElementById('cleanupStats').textContent =
                    `模型文件: ${c.models.count} 个 / ${fmtSize(c.models.size)} | 上传数据: ${c.uploads.count} 个 / ${fmtSize(c.uploads.size)} | 日志: ${c.logs.count} 个 / ${fmtSize(c.logs.size)}`;
            } catch (e) { console.error('cleanup status failed:', e); }
        }

        async function setRetentionDays() {
            const days = parseInt(document.getElementById('retentionDaysInput').value || '0', 10);
            await fetch('/admin/api/cleanup/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ retention_days: days })
            });
            alert(days > 0 ? `自动清理已开启：保留 ${days} 天` : '自动清理已关闭');
            loadCleanupStatus();
        }

        async function runCleanup(targets) {
            const names = { models: '模型文件', uploads: '上传数据', logs: '日志' };
            if (!confirm('确认清除：' + targets.map(t => names[t]).join('、') + '？此操作不可恢复')) return;
            try {
                const res = await fetch('/admin/api/cleanup/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ targets })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    alert(`清理完成，释放 ${data.freed_mb} MB`);
                    loadCleanupStatus();
                } else {
                    alert(data.message || '清理失败');
                }
            } catch (e) { alert('清理请求失败'); }
        }

        // 实时监控（CPU/RAM 折线图）
        let monitorInterval = null;
        const MAX_POINTS = 60;
        let cpuData = [], ramData = [];

        // Catmull-Rom 转 Bezier 平滑曲线（定义在外部，避免重复声明）
        function smoothCurve(pts, ctx, close) {
            if (pts.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(pts[0].x, pts[0].y);
            if (pts.length === 2) { ctx.lineTo(pts[1].x, pts[1].y); return; }
            const t = 0.3;
            for (let i = 0; i < pts.length - 1; i++) {
                const p0 = pts[i === 0 ? 0 : i - 1];
                const p1 = pts[i];
                const p2 = pts[i + 1];
                const p3 = pts[i + 2 >= pts.length ? pts.length - 1 : i + 2];
                ctx.bezierCurveTo(
                    p1.x + (p2.x - p0.x) * t, p1.y + (p2.y - p0.y) * t,
                    p2.x - (p3.x - p1.x) * t, p2.y - (p3.y - p1.y) * t,
                    p2.x, p2.y
                );
            }
            if (close) ctx.closePath();
        }

        function openMonitor() {
            document.getElementById('monitorModal').classList.add('show');
            cpuData = []; ramData = [];
            if (monitorInterval) clearInterval(monitorInterval);
            fetchMonitorData();
            monitorInterval = setInterval(fetchMonitorData, 1000);
        }

        function closeMonitor() {
            document.getElementById('monitorModal').classList.remove('show');
            if (monitorInterval) { clearInterval(monitorInterval); monitorInterval = null; }
        }

        async function fetchMonitorData() {
            try {
                const res = await fetch('/admin/api/monitor');
                const data = await res.json();
                if (data.status !== 'success') return;
                const now = new Date();
                cpuData.push({ time: now, value: data.cpu.usage });
                ramData.push({ time: now, value: data.ram.usage });
                if (cpuData.length > MAX_POINTS) cpuData.shift();
                if (ramData.length > MAX_POINTS) ramData.shift();
                document.getElementById('liveCpu').textContent = data.cpu.usage + '%';
                document.getElementById('liveRam').textContent = data.ram.usage + '%';
                drawChart('cpuChart', cpuData, '#ef4444', 'CPU %');
                drawChart('ramChart', ramData, '#818cf8', 'RAM %');
            } catch (e) { console.error('Monitor fetch failed:', e); }
        }

        function drawChart(canvasId, data, color, label) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = (rect.width || 600) * dpr;
            canvas.height = (rect.height || 200) * dpr;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);
            const w = rect.width || 600, h = rect.height || 200;
            ctx.clearRect(0, 0, w, h);

            if (data.length < 2) {
                ctx.fillStyle = 'rgba(255,255,255,0.3)';
                ctx.font = '14px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('等待数据...', w / 2, h / 2);
                return;
            }

            const pad = { top: 10, bottom: 20, left: 35, right: 10 };
            const cw = w - pad.left - pad.right, ch = h - pad.top - pad.bottom;
            const minVal = Math.max(0, Math.min(...data.map(d => d.value)) - 5);
            const maxVal = Math.min(100, Math.max(...data.map(d => d.value)) + 5);
            const range = maxVal - minVal || 1;

            // 数据点转画布坐标（按实际点数铺满横轴，避免数据不足 MAX_POINTS 时挤在左侧）
            const xDenom = Math.max(data.length - 1, 1);
            const pts = data.map((d, i) => ({
                x: pad.left + (i / xDenom) * cw,
                y: pad.top + (1 - (d.value - minVal) / range) * ch
            }));

            // Grid
            ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = pad.top + (ch / 4) * i;
                ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
                ctx.fillStyle = 'rgba(255,255,255,0.2)';
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText((maxVal - (range / 4) * i).toFixed(0), pad.left - 5, y + 3);
            }

// 填充区域（复用平滑曲线路径，保证填充边界与描边一致）
            smoothCurve(pts, ctx, false);
            ctx.lineTo(pts[pts.length - 1].x, pad.top + ch);
            ctx.lineTo(pts[0].x, pad.top + ch);
            ctx.closePath();
            const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
            grad.addColorStop(0, color + '40');
            grad.addColorStop(1, color + '05');
            ctx.fillStyle = grad;
            ctx.fill();

            // 曲线（平滑）
            smoothCurve(pts, ctx, false);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // 曲线发光
            smoothCurve(pts, ctx, false);
            ctx.strokeStyle = color + '20';
            ctx.lineWidth = 6;
            ctx.stroke();

            // 当前点
            const last = pts[pts.length - 1];
            ctx.beginPath(); ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#fff'; ctx.fill();
            ctx.beginPath(); ctx.arc(last.x, last.y, 7, 0, Math.PI * 2);
            ctx.strokeStyle = color + '80'; ctx.lineWidth = 2; ctx.stroke();
        }

        // 设备状态
        async function loadDeviceStatus() {
            try {
                const res = await fetch('/admin/api/device_status');
                const data = await res.json();
                if (data.status !== 'success') return;
                const d = data;
                document.getElementById('devOs').textContent = d.os;
                document.getElementById('devPython').textContent = d.python;
                document.getElementById('cpuModel').textContent = d.cpu.model;
                document.getElementById('cpuUsage').textContent = d.cpu.usage + '%';
                document.getElementById('cpuFreq').textContent = d.cpu.freq + ' MHz';
                document.getElementById('ramTotal').textContent = d.ram.total + ' GB';
                document.getElementById('ramUsed').textContent = d.ram.used + ' GB';
                document.getElementById('ramAvail').textContent = d.ram.available + ' GB';
                document.getElementById('ramUsage').textContent = d.ram.usage + '%';
                document.getElementById('diskTotal').textContent = d.disk.total + ' GB';
                document.getElementById('diskUsed').textContent = d.disk.used + ' GB';
                document.getElementById('diskFree').textContent = d.disk.free + ' GB';
                document.getElementById('diskUsage').textContent = d.disk.usage + '%';
                if (d.gpu.supported) {
                    document.getElementById('gpuName').textContent = d.gpu.info.name;
                    document.getElementById('gpuMem').textContent = d.gpu.info.memory_allocated + ' GB / ' + d.gpu.info.memory_reserved + ' GB';
                    document.getElementById('gpuSupport').textContent = '✅ 支持';
                    document.getElementById('gpuSupport').style.color = 'var(--success)';
                    document.getElementById('gpuNote').textContent = '';
                }
                if (d.xpu.supported) {
                    document.getElementById('xpuSupport').textContent = '✅ 支持';
                    document.getElementById('xpuSupport').style.color = 'var(--success)';
                    document.getElementById('xpuNote').textContent = '';
                }
                if (d.npu.supported) {
                    document.getElementById('npuSupport').textContent = '✅ 支持';
                    document.getElementById('npuSupport').style.color = 'var(--success)';
                    document.getElementById('npuNote').textContent = '';
                }
            } catch (e) {
                console.error('Device status load failed:', e);
            }
        }

        // 训练队列
        async function loadQueue() {
            try {
                const res = await fetch('/admin/api/queue');
                const data = await res.json();
                if (data.status !== 'success') return;
                const q = data.queue;
                document.getElementById('queueActive').textContent = q.active_count;
                document.getElementById('queueWaiting').textContent = q.queue_length;
                document.getElementById('maxConcurrentInput').value = q.max_concurrent;
                const tbody = document.getElementById('queueTableBody');
                const items = [...q.active, ...q.waiting];
                if (items.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">暂无训练任务</td></tr>';
                    return;
                }
                const iconMap = { 'image': '🖼️', 'text': '📝' };
                tbody.innerHTML = items.map(item => {
                    const type = (item.params && item.params.train_type) || '?';
                    const isActive = q.active.some(a => a.task_id === item.task_id);
                    const statusText = isActive ? '🔄 运行中' : '⏳ 等待中';
                    const statusColor = isActive ? 'var(--success)' : 'var(--warning)';
                    const progress = item.progress || 0;
                    const paramsStr = item.params ? JSON.stringify(item.params) : '{}';
                    return `<tr>
                        <td style="font-family:monospace;font-size:13px;">${item.task_id}</td>
                        <td>${item.user_id}</td>
                        <td>${iconMap[type] || '📦'} ${type}</td>
                        <td><span style="color:${statusColor};">${statusText}</span></td>
                        <td>${isActive ? progress + '%' : '—'}</td>
                        <td>
                            <button class="btn-sm btn-danger" onclick="cancelTask('${item.task_id}')">🛑 停止</button>
                        </td>
                    </tr>`;
                }).join('');
            } catch (e) {
                document.getElementById('queueTableBody').innerHTML = '<tr><td colspan="6" class="empty-state">❌ 加载失败</td></tr>';
            }
        }

        async function setMaxConcurrent() {
            const n = parseInt(document.getElementById('maxConcurrentInput').value) || 5;
            try {
                const res = await fetch('/admin/api/queue/max_concurrent', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_concurrent: n })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast('✅ 最大并发数已更新', 'success');
                    loadQueue();
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        async function cancelTask(taskId) {
            if (!confirm(`确定要停止/取消任务 ${taskId} 吗？`)) return;
            try {
                const res = await fetch(`/admin/api/queue/cancel/${taskId}`, { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast('🛑 任务已停止', 'success');
                    loadQueue();
                } else {
                    showToast('❌ ' + data.message, 'error');
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        // 初始化轮询队列
        setInterval(loadQueue, 3000);

        // 带宽管理
        async function loadBandwidth() {
            try {
                const res = await fetch('/admin/api/bandwidth');
                const data = await res.json();
                if (data.status !== 'success') return;
                const bw = data.bandwidth;
                document.getElementById('defaultBandwidthInput').value = bw.default_limit_mbps;
                const tbody = document.getElementById('bandwidthTableBody');
                if (!bw.users || bw.users.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="empty-state">暂无带宽数据</td></tr>';
                    return;
                }
                tbody.innerHTML = bw.users.map(u => {
                    const status = u.exceeded
                        ? '<span style="color:var(--error);">⚠️ 超限</span>'
                        : '<span style="color:var(--success);">✅ 正常</span>';
                    return `<tr>
                        <td>${u.user_id}</td>
                        <td>${u.username || u.user_id}</td>
                        <td>${u.current_mbps} Mbps</td>
                        <td>
                            <input type="number" class="select-group" style="width:80px" value="${u.limit_mbps}" min="0.1" max="1000" step="0.1"
                                   id="bwLimit_${u.user_id}">
                        </td>
                        <td>${status}</td>
                        <td><button class="btn-sm" onclick="setUserBandwidth('${u.user_id}')">💾 保存</button></td>
                    </tr>`;
                }).join('');
            } catch (e) {
                document.getElementById('bandwidthTableBody').innerHTML = '<tr><td colspan="5" class="empty-state">❌ 加载失败</td></tr>';
            }
        }

        async function setDefaultBandwidth() {
            const mbps = parseFloat(document.getElementById('defaultBandwidthInput').value) || 10;
            try {
                const res = await fetch('/admin/api/bandwidth/default', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mbps })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast('✅ 默认带宽已更新', 'success');
                    loadBandwidth();
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        async function setUserBandwidth(userId) {
            const input = document.getElementById('bwLimit_' + userId);
            const mbps = parseFloat(input ? input.value : 10) || 10;
            try {
                const res = await fetch(`/admin/api/bandwidth/user/${userId}`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mbps })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    showToast('✅ 用户带宽已更新', 'success');
                    loadBandwidth();
                }
            } catch (e) { showToast('❌ 操作失败', 'error'); }
        }

        setInterval(loadBandwidth, 3000);
