// homepage.js —— 由模板 homepage.html 的内联脚本迁出（保持原执行顺序）
        // 粒子背景
        const canvas = document.getElementById('particles-canvas');
        const ctx = canvas.getContext('2d');
        let particles = [], animId;

        function resize() {
            canvas.width = window.innerWidth;
            canvas.height = document.body.scrollHeight;
        }
        function createParticles(count) {
            particles = [];
            for (let i = 0; i < count; i++) {
                particles.push({
                    x: Math.random() * canvas.width, y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
                    r: Math.random() * 2 + 0.5, alpha: Math.random() * 0.3 + 0.1
                });
            }
        }
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0) p.x = canvas.width; if (p.x > canvas.width) p.x = 0;
                if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;
                ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(99, 102, 241, ${p.alpha})`; ctx.fill();
            });
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120) {
                        ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y); ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(99, 102, 241, ${0.06 * (1 - dist / 120)})`;
                        ctx.lineWidth = 0.5; ctx.stroke();
                    }
                }
            }
            animId = requestAnimationFrame(draw);
        }
        window.addEventListener('resize', () => { resize(); createParticles(60); });
        resize(); createParticles(60); draw();

        // 获取版本号
        fetch('/api/version')
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success' && data.version) {
                    const v = data.version;
                    document.getElementById('versionBadge').textContent = `🚀 v${v.ResVersion} · 统一训练平台`;
                    const fv = document.getElementById('footerVersion');
                    if (fv) fv.textContent = v.ResVersion;
                }
            })
            .catch(() => {
                document.getElementById('versionBadge').textContent = '🚀 统一训练平台';
            });
