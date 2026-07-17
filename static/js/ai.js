// ================================= 可配置项（开发者可自由修改） =================================
const CONFIG = {
    tokenDemoText: "我喜欢吃苹果",
    embeddingWords: [
        {word: "猫", x: 100, y: 150},
        {word: "狗", x: 130, y: 170},
        {word: "汽车", x: 500, y: 300},
        {word: "苹果", x: 120, y: 300},
        {word: "香蕉", x: 150, y: 280},
        {word: "飞机", x: 550, y: 100}
    ],
    demoSentence: ["我", "爱", "人工智能"],
    attentionWeights: {
        "我": {"我": 0.2, "爱": 0.5, "人工智能": 0.3},
        "爱": {"我": 0.3, "爱": 0.2, "人工智能": 0.5},
        "人工智能": {"我": 0.2, "爱": 0.5, "人工智能": 0.3}
    }
};

// ================================= 页面初始化 =================================
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initScrollAnimation();
    initNavHighlight();
    initSmoothScroll();
    initTokenDemo();
    initEmbeddingDemo();
    initAttentionDemo();
    initLossDemo();
    initGradientDemo();
    initTransformerTooltip();
});

// ================================= 0. 粒子背景 =================================
function initParticles() {
    const canvas = document.createElement('canvas');
    canvas.id = 'particles-canvas';
    document.body.prepend(canvas);
    const ctx = canvas.getContext('2d');
    let particles = [];
    let animId = null;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = document.body.scrollHeight;
    }

    function createParticles(count) {
        particles = [];
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                r: Math.random() * 2.5 + 0.5,
                alpha: Math.random() * 0.4 + 0.1
            });
        }
    }

    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(99, 102, 241, ${p.alpha})`;
            ctx.fill();
        });
        // 连线（相邻粒子）
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${0.08 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        animId = requestAnimationFrame(drawParticles);
    }

    window.addEventListener('resize', () => {
        resize();
        createParticles(80);
    });

    resize();
    createParticles(80);
    drawParticles();
}

// ================================= 滚动淡入动画 =================================
function initScrollAnimation() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.05, rootMargin: '0px 0px -30px 0px' });
    document.querySelectorAll('section').forEach(sec => observer.observe(sec));
}

// ================================= 导航栏滚动高亮 =================================
function initNavHighlight() {
    window.addEventListener('scroll', () => {
        const sections = document.querySelectorAll('section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        let current = '';
        sections.forEach(sec => {
            const secTop = sec.offsetTop;
            if (pageYOffset >= secTop - 200) {
                current = sec.getAttribute('id');
            }
        });
        navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href').slice(1) === current);
        });
    });
}

// ================================= 平滑滚动 =================================
function initSmoothScroll() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ================================= 1. 分词交互逻辑（BPE 分词器本地实现）=================================
let _bpeTokenizer = null;

async function getBpeTokenizer() {
    if (_bpeTokenizer) return _bpeTokenizer;
    const res = await fetch('/static/tokenizer_bpe.json');
    const data = await res.json();
    _bpeTokenizer = {
        vocab: new Map(Object.entries(data.vocab).map(([k, v]) => [k, v])),
        merges: data.merges.map(m => m.split(' ')),
        cache: new Map()
    };
    return _bpeTokenizer;
}

function bpeTokenize(text, tokenizer) {
    const cache = tokenizer.cache;
    if (cache.has(text)) return cache.get(text);

    // 1. 预分词：按空格和标点拆分
    const words = text.match(/[a-zA-Z]+|\d+|[^\s]/g) || [text];
    const result = [];

    for (const word of words) {
        if (tokenizer.vocab.has(word)) {
            result.push({ text: word, id: tokenizer.vocab.get(word) });
            continue;
        }

        // 2. 拆成字符级 token
        let symbols = word.split('').map(c => c);
        
        // 3. 应用 BPE merge 规则
        let changed = true;
        while (changed) {
            changed = false;
            let bestPair = null;
            let bestRank = Infinity;

            for (let i = 0; i < symbols.length - 1; i++) {
                const pair = symbols[i] + ' ' + symbols[i + 1];
                const rank = tokenizer.merges.findIndex(m => m[0] === symbols[i] && m[1] === symbols[i + 1]);
                if (rank >= 0 && rank < bestRank) {
                    bestRank = rank;
                    bestPair = i;
                }
            }

            if (bestPair !== null) {
                const merged = symbols[bestPair] + symbols[bestPair + 1];
                symbols.splice(bestPair, 2, merged);
                changed = true;
            }
        }

        // 4. 映射到 vocab ID
        for (const s of symbols) {
            const id = tokenizer.vocab.get(s);
            if (id !== undefined) {
                result.push({ text: s, id });
            } else {
                // 未知字符：按字符拆
                for (const ch of s) {
                    result.push({ text: ch, id: tokenizer.vocab.get(ch) ?? -1 });
                }
            }
        }
    }

    cache.set(text, result);
    return result;
}

function initTokenDemo() {
    const tokenInput = document.getElementById('tokenInput');
    const tokenBtn = document.getElementById('tokenBtn');
    const tokenDisplay = document.getElementById('tokenDisplay');

    async function updateTokenDemo() {
        const text = tokenInput.value || CONFIG.tokenDemoText;
        tokenDisplay.innerHTML = '<span class="token" style="background:transparent;color:var(--text-muted);font-size:0.8rem">⏳ 加载分词器中...</span>';

        try {
            const tokenizer = await getBpeTokenizer();
            tokenDisplay.innerHTML = '<span class="token" style="background:transparent;color:var(--text-muted);font-size:0.8rem">⏳ 分词中...</span>';

            // 使用 requestAnimationFrame 让浏览器有时间渲染 loading 状态
            await new Promise(r => requestAnimationFrame(r));

            const tokens = bpeTokenize(text, tokenizer);

            tokenDisplay.innerHTML = '';
            tokens.forEach((token, index) => {
                const span = document.createElement('span');
                span.className = 'token';
                span.textContent = token.text;
                span.title = `Token ID: ${token.id} | 位置: ${index}`;
                span.style.animationDelay = `${index * 0.06}s`;
                const hue = (token.id * 37) % 360;
                span.style.background = `linear-gradient(135deg, hsl(${hue}, 70%, 55%), hsl(${(hue + 30) % 360}, 70%, 45%))`;
                tokenDisplay.appendChild(span);
            });
        } catch (err) {
            tokenDisplay.innerHTML = `<span class="token" style="background:transparent;color:var(--error)">❌ 分词失败: ${err.message}</span>`;
        }
    }

    tokenBtn.addEventListener('click', updateTokenDemo);
    tokenInput.addEventListener('input', updateTokenDemo);
    tokenInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') updateTokenDemo();
    });
    updateTokenDemo();
}

// ================================= 2. 词嵌入可视化逻辑 =================================
function initEmbeddingDemo() {
    const embeddingCanvas = document.getElementById('embeddingCanvas');
    const embeddingCtx = embeddingCanvas.getContext('2d');
    const tooltip = document.getElementById('embeddingTooltip');
    const newWordInput = document.getElementById('newWordInput');
    const addWordBtn = document.getElementById('addWordBtn');
    let words = CONFIG.embeddingWords.map(w => ({...w}));
    let selectedWord = null;

    function resizeCanvas() {
        const rect = embeddingCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = rect.width || 700;
        const h = rect.height || 400;
        embeddingCanvas.width = w * dpr;
        embeddingCanvas.height = h * dpr;
        embeddingCtx.scale(dpr, dpr);
        return { w, h, dpr };
    }

    function drawEmbedding() {
        const { w, h } = resizeCanvas();
        // 清空（背景由CSS提供）
        embeddingCtx.clearRect(0, 0, w, h);

        // 绘制网格
        embeddingCtx.strokeStyle = 'rgba(255,255,255,0.05)';
        embeddingCtx.lineWidth = 1;
        for (let i = 0; i < w; i += 50) {
            embeddingCtx.beginPath();
            embeddingCtx.moveTo(i, 0);
            embeddingCtx.lineTo(i, h);
            embeddingCtx.stroke();
        }
        for (let i = 0; i < h; i += 50) {
            embeddingCtx.beginPath();
            embeddingCtx.moveTo(0, i);
            embeddingCtx.lineTo(w, i);
            embeddingCtx.stroke();
        }

        // 绘制连接线（选中词到其他词）
        if (selectedWord) {
            words.forEach(word => {
                if (word === selectedWord) return;
                const dist = Math.sqrt((word.x - selectedWord.x) ** 2 + (word.y - selectedWord.y) ** 2);
                const sim = Math.max(0, 1 - dist / 350);
                embeddingCtx.beginPath();
                embeddingCtx.moveTo(selectedWord.x, selectedWord.y);
                embeddingCtx.lineTo(word.x, word.y);
                embeddingCtx.strokeStyle = `rgba(99, 102, 241, ${sim * 0.3})`;
                embeddingCtx.lineWidth = sim * 2 + 0.5;
                embeddingCtx.stroke();
            });
        }

        // 绘制词汇点
        words.forEach(word => {
            const isSelected = word === selectedWord;
            const radius = isSelected ? 16 : 12;
            const gradient = embeddingCtx.createRadialGradient(word.x, word.y, 0, word.x, word.y, radius * 2);
            gradient.addColorStop(0, isSelected ? '#818cf8' : '#6366f1');
            gradient.addColorStop(1, isSelected ? '#6366f1' : '#4f46e5');

            embeddingCtx.beginPath();
            embeddingCtx.arc(word.x, word.y, radius, 0, Math.PI * 2);
            embeddingCtx.fillStyle = gradient;
            embeddingCtx.fill();
            embeddingCtx.strokeStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.3)';
            embeddingCtx.lineWidth = isSelected ? 3 : 1.5;
            embeddingCtx.stroke();

            // 发光效果
            if (isSelected) {
                embeddingCtx.beginPath();
                embeddingCtx.arc(word.x, word.y, radius + 6, 0, Math.PI * 2);
                embeddingCtx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
                embeddingCtx.lineWidth = 2;
                embeddingCtx.stroke();
            }

            // 文字
            embeddingCtx.fillStyle = '#f1f5f9';
            embeddingCtx.font = isSelected ? 'bold 15px Inter, sans-serif' : '14px Inter, sans-serif';
            embeddingCtx.textAlign = 'center';
            embeddingCtx.textBaseline = 'bottom';
            embeddingCtx.fillText(word.word, word.x, word.y - radius - 4);
        });
    }

    // 悬停检测
    function getWordAt(x, y) {
        for (let i = words.length - 1; i >= 0; i--) {
            const w = words[i];
            const dist = Math.sqrt((x - w.x) ** 2 + (y - w.y) ** 2);
            if (dist < 20) return w;
        }
        return null;
    }

    embeddingCanvas.addEventListener('mousemove', (e) => {
        const rect = embeddingCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const word = getWordAt(x, y);

        if (word) {
            tooltip.style.opacity = 1;
            tooltip.style.left = `${e.clientX + 12}px`;
            tooltip.style.top = `${e.clientY + 12}px`;
            const similarities = words.filter(w => w !== word).map(w => {
                const dist = Math.sqrt((w.x - word.x) ** 2 + (w.y - word.y) ** 2);
                return { word: w.word, sim: Math.max(0, (1 - dist / 400)).toFixed(2) };
            }).sort((a, b) => b.sim - a.sim).slice(0, 3);
            tooltip.innerHTML = `<strong>「${word.word}」</strong><br>最近邻：<br>${similarities.map(s => `　${s.word}: ${(parseFloat(s.sim) * 100).toFixed(0)}%`).join('<br>')}`;
            embeddingCanvas.style.cursor = 'pointer';
        } else {
            tooltip.style.opacity = 0;
            embeddingCanvas.style.cursor = 'default';
        }
    });

    embeddingCanvas.addEventListener('mouseleave', () => {
        tooltip.style.opacity = 0;
    });

    embeddingCanvas.addEventListener('click', (e) => {
        const rect = embeddingCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const word = getWordAt(x, y);
        if (word) {
            selectedWord = (selectedWord === word) ? null : word;
            drawEmbedding();
        }
    });

    // 添加新词
    addWordBtn.addEventListener('click', () => {
        const newWord = newWordInput.value.trim();
        if (!newWord) return;
        const rect = embeddingCanvas.getBoundingClientRect();
        const w = rect.width || 700;
        const h = rect.height || 400;
        const x = 60 + Math.random() * (w - 120);
        const y = 60 + Math.random() * (h - 120);
        words.push({ word: newWord, x, y });
        newWordInput.value = '';
        drawEmbedding();
    });

    newWordInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') addWordBtn.click();
    });

    drawEmbedding();
    window.addEventListener('resize', drawEmbedding);
}

// ================================= 3. 自注意力交互逻辑 =================================
let attentionInitialized = false;

function initAttentionDemo() {
    const sentenceDisplay = document.getElementById('sentenceDisplay');
    const attentionChart = document.getElementById('attentionChart');
    const detailDiv = document.getElementById('attentionDetail');

    CONFIG.demoSentence.forEach((word) => {
        const wordEl = document.createElement('div');
        wordEl.className = 'word-token';
        wordEl.textContent = word;
        wordEl.addEventListener('click', () => showAttention(word, wordEl));
        sentenceDisplay.appendChild(wordEl);
    });

    function renderBars(targetWord) {
        attentionChart.innerHTML = '';
        const weights = CONFIG.attentionWeights[targetWord];
        CONFIG.demoSentence.forEach((word, index) => {
            const weight = weights[word];
            const bar = document.createElement('div');
            bar.className = 'attention-bar';
            bar.style.height = `${Math.max(weight * 100, 5)}%`;
            bar.innerHTML = `<span>${word}</span><div class="weight">${(weight * 100).toFixed(0)}%</div>`;
            bar.style.animation = `popIn 0.4s ease ${index * 0.08}s both`;
            // 不同颜色深浅
            const alpha = 0.4 + weight * 0.6;
            bar.style.background = `linear-gradient(to top, rgba(99, 102, 241, ${alpha}), rgba(129, 140, 248, ${alpha * 0.8}))`;
            attentionChart.appendChild(bar);
        });
    }

    function showAttention(targetWord, el) {
        document.querySelectorAll('.word-token').forEach(w => w.classList.remove('active'));
        el.classList.add('active');
        renderBars(targetWord);
        detailDiv.innerHTML = `
            <strong>🎯 当前选中词：${targetWord}</strong>
            <p style="margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.9rem;">
                <strong style="color: var(--accent-hover);">Q（Query）：</strong> 「${targetWord}」的查询向量主动查询其他词<br>
                <strong style="color: var(--accent-hover);">K（Key）：</strong> 其他词提供 Key 向量与 Query 计算相似度<br>
                <strong style="color: var(--accent-hover);">V（Value）：</strong> 用权重加权所有 Value，得到上下文感知表示
            </p>
            <p style="margin-top: 0.5rem; color: var(--text-muted); font-size: 0.8rem;">
                💡 上方柱状图高度表示「${targetWord}」对其他词的注意力分配比例
            </p>`;
    }

    // 默认选中第一个
    const first = sentenceDisplay.querySelector('.word-token');
    if (first) {
        setTimeout(() => {
            showAttention(CONFIG.demoSentence[0], first);
        }, 300);
    }
}

// ================================= 4. 梯度下降演示逻辑 =================================
function initGradientDemo() {
    const gradientCanvas = document.getElementById('gradientCanvas');
    const gradientCtx = gradientCanvas.getContext('2d');
    const lrSlider = document.getElementById('lrSlider');
    const lrValue = document.getElementById('lrValue');
    const resetBtn = document.getElementById('resetGradientBtn');

    let lr = 0.1;
    let ball = { x: 5, y: 5 };
    let animationId = null;
    let trail = [];
    let isFirstDraw = true;

    function setupCanvas() {
        const rect = gradientCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = rect.width || 700;
        const h = rect.height || 400;
        gradientCanvas.width = w * dpr;
        gradientCanvas.height = h * dpr;
        gradientCtx.scale(dpr, dpr);
        return { w, h };
    }

    function drawLossContour(w, h) {
        // 绘制等高线（先画，不清除）
        for (let i = 0; i < 25; i++) {
            const r = 10 + i * 12;
            gradientCtx.beginPath();
            gradientCtx.arc(w / 2, h / 2, r, 0, Math.PI * 2);
            gradientCtx.strokeStyle = `rgba(99, 102, 241, ${0.06 + i * 0.015})`;
            gradientCtx.lineWidth = 1;
            gradientCtx.stroke();
        }

        // 最低点标记
        const centerX = w / 2, centerY = h / 2;
        gradientCtx.beginPath();
        gradientCtx.arc(centerX, centerY, 6, 0, Math.PI * 2);
        gradientCtx.fillStyle = '#22c55e';
        gradientCtx.fill();

        // 脉冲光环
        gradientCtx.beginPath();
        gradientCtx.arc(centerX, centerY, 10 + (Date.now() * 0.002) % 8, 0, Math.PI * 2);
        gradientCtx.strokeStyle = 'rgba(34, 197, 94, 0.3)';
        gradientCtx.lineWidth = 1.5;
        gradientCtx.stroke();

        gradientCtx.fillStyle = '#94a3b8';
        gradientCtx.font = '12px Inter, sans-serif';
        gradientCtx.textAlign = 'center';
        gradientCtx.fillText('★ 最优点', centerX, centerY - 15);
    }

    function animateGradient() {
        const { w, h } = setupCanvas();
        const centerX = w / 2;
        const centerY = h / 2;
        const scale = 10;

        // 梯度 = -当前位置向量（指向中心）
        const gx = -ball.x;
        const gy = -ball.y;
        const gradNorm = Math.sqrt(gx * gx + gy * gy);

        // 更新位置
        ball.x += lr * gx;
        ball.y += lr * gy;

        // 记录轨迹
        trail.push({ x: centerX + ball.x * scale, y: centerY + ball.y * scale });
        if (trail.length > 150) trail.shift();

        // ===== 完整绘制帧 =====
        gradientCtx.clearRect(0, 0, w, h);

        // 等高线
        drawLossContour(w, h);

        // 轨迹路径
        if (trail.length > 1) {
            gradientCtx.beginPath();
            gradientCtx.moveTo(trail[0].x, trail[0].y);
            for (let i = 1; i < trail.length; i++) {
                gradientCtx.lineTo(trail[i].x, trail[i].y);
            }
            gradientCtx.strokeStyle = '#ef4444';
            gradientCtx.lineWidth = 2;
            gradientCtx.stroke();

            // 轨迹发光
            gradientCtx.beginPath();
            gradientCtx.moveTo(trail[0].x, trail[0].y);
            for (let i = 1; i < trail.length; i++) {
                gradientCtx.lineTo(trail[i].x, trail[i].y);
            }
            gradientCtx.strokeStyle = 'rgba(239, 68, 68, 0.15)';
            gradientCtx.lineWidth = 6;
            gradientCtx.stroke();
        }

        // 梯度向量（橙色箭头）
        if (gradNorm > 0.01) {
            const arrowLen = Math.min(gradNorm * scale * 0.4, 60);
            const angle = Math.atan2(gy, gx);
            const fromX = centerX + ball.x * scale;
            const fromY = centerY + ball.y * scale;
            const toX = fromX + Math.cos(angle) * arrowLen;
            const toY = fromY + Math.sin(angle) * arrowLen;

            gradientCtx.beginPath();
            gradientCtx.moveTo(fromX, fromY);
            gradientCtx.lineTo(toX, toY);
            gradientCtx.strokeStyle = '#f59e0b';
            gradientCtx.lineWidth = 3;
            gradientCtx.stroke();

            // 箭头头部
            const headLen = 10;
            gradientCtx.beginPath();
            gradientCtx.moveTo(toX, toY);
            gradientCtx.lineTo(toX - headLen * Math.cos(angle - 0.4), toY - headLen * Math.sin(angle - 0.4));
            gradientCtx.moveTo(toX, toY);
            gradientCtx.lineTo(toX - headLen * Math.cos(angle + 0.4), toY - headLen * Math.sin(angle + 0.4));
            gradientCtx.stroke();

            // 梯度标签
            gradientCtx.fillStyle = '#f59e0b';
            gradientCtx.font = '11px Inter, sans-serif';
            gradientCtx.textAlign = 'center';
            gradientCtx.fillText('梯度方向 ↗', toX, toY - 12);
        }

        // 参数小球
        const ballX = centerX + ball.x * scale;
        const ballY = centerY + ball.y * scale;

        // 球体发光
        gradientCtx.beginPath();
        gradientCtx.arc(ballX, ballY, 14, 0, Math.PI * 2);
        gradientCtx.fillStyle = 'rgba(239, 68, 68, 0.15)';
        gradientCtx.fill();

        // 球体
        gradientCtx.beginPath();
        gradientCtx.arc(ballX, ballY, 9, 0, Math.PI * 2);
        gradientCtx.fillStyle = '#ef4444';
        gradientCtx.fill();
        gradientCtx.strokeStyle = 'rgba(255,255,255,0.6)';
        gradientCtx.lineWidth = 2;
        gradientCtx.stroke();

        // 球体高光
        gradientCtx.beginPath();
        gradientCtx.arc(ballX - 3, ballY - 3, 3, 0, Math.PI * 2);
        gradientCtx.fillStyle = 'rgba(255,255,255,0.3)';
        gradientCtx.fill();

        // 显示当前信息
        gradientCtx.fillStyle = '#94a3b8';
        gradientCtx.font = '12px Inter, sans-serif';
        gradientCtx.textAlign = 'left';
        gradientCtx.fillText(`当前位置: (${ball.x.toFixed(2)}, ${ball.y.toFixed(2)})`, 12, 20);
        gradientCtx.fillText(`梯度范数: ${gradNorm.toFixed(3)}`, 12, 38);
        gradientCtx.fillText(`学习率: ${lr}`, 12, 56);

        // 继续或停止
        if (gradNorm > 0.02 && !isNaN(ball.x) && !isNaN(ball.y)) {
            animationId = requestAnimationFrame(animateGradient);
        } else {
            // 到达最低点
            gradientCtx.fillStyle = '#22c55e';
            gradientCtx.font = 'bold 14px Inter, sans-serif';
            gradientCtx.textAlign = 'center';
            gradientCtx.fillText('✅ 已收敛到最优点！', w / 2, 30);
        }
    }

    // 学习率调整
    lrSlider.addEventListener('input', () => {
        lr = parseFloat(lrSlider.value);
        lrValue.textContent = lr.toFixed(2);
    });

    // 重置动画
    function resetGradient() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        ball = { x: 5, y: 5 };
        trail = [];
        setTimeout(() => animateGradient(), 50);
    }

    resetBtn.addEventListener('click', resetGradient);

    // 按回车重置
    document.addEventListener('keydown', (e) => {
        if (e.key === 'r' || e.key === 'R') resetGradient();
    });

    // 初始化
    window.addEventListener('resize', () => {
        if (!animationId) {
            const { w, h } = setupCanvas();
            gradientCtx.clearRect(0, 0, w, h);
            drawLossContour(w, h);
        }
    });

    setTimeout(resetGradient, 200);
}

// ================================= 5. Transformer 架构 tooltip =================================
function initTransformerTooltip() {
    const descriptions = {
        "📝 输入文本": "将原始文本拆分为 Token 序列，是模型处理的起点",
        "🔢 词嵌入": "将每个 Token 映射为高维语义向量，捕捉词汇含义与上下文关系",
        "🧠 多层注意力": "通过自注意力机制学习 Token 之间的依赖关系，多层堆叠提取深层语义特征",
        "📤 输出结果": "模型基于学习到的表示生成最终的文本、分类或推理结果"
    };

    document.querySelectorAll('.flow-step').forEach(step => {
        const textEl = step.querySelector('div:first-child');
        if (!textEl) return;
        step.addEventListener('mouseenter', function () {
            const text = textEl.textContent;
            const desc = descriptions[text];
            if (desc) {
                this.title = desc;
            }
            this.style.cursor = 'help';
        });
    });
}

// ================================= 6. 损失率与困惑度交互演示 =================================
function initLossDemo() {
    const canvas = document.getElementById('lossCanvas');
    const ctx = canvas.getContext('2d');
    const lrSlider = document.getElementById('lossLrSlider');
    const lrValue = document.getElementById('lossLrValue');
    const noiseSlider = document.getElementById('noiseSlider');
    const noiseValue = document.getElementById('noiseValue');
    const resetBtn = document.getElementById('resetLossBtn');
    const currentLossEl = document.getElementById('currentLoss');
    const currentPplEl = document.getElementById('currentPpl');
    const epochCountEl = document.getElementById('epochCount');

    let lr = 0.01, noise = 0.05;
    let epochs = 0, maxEpochs = 200;
    let lossHistory = [];
    let animId = null;

    function setupCanvas() {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = rect.width || 700, h = rect.height || 350;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);
        return { w, h };
    }

    function simulateLossStep(currentLoss) {
        const gradient = currentLoss * 0.3;
        const decay = lr * gradient;
        const randomNoise = (Math.random() - 0.5) * noise * 2;
        return Math.max(currentLoss - decay + randomNoise, 0.001);
    }

    function drawLossChart(w, h) {
        ctx.clearRect(0, 0, w, h);
        const pad = { top: 20, bottom: 30, left: 50, right: 20 };
        const chartW = w - pad.left - pad.right;
        const chartH = h - pad.top - pad.bottom;

        // 网格
        ctx.strokeStyle = 'rgba(255,255,255,0.04)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (chartH / 5) * i;
            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        }

        // Y轴
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (let i = 0; i <= 5; i++) {
            ctx.fillText((5 - i).toFixed(1), pad.left - 8, pad.top + (chartH / 5) * i);
        }
        // X轴
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        for (let i = 0; i <= 5; i++) {
            ctx.fillText(Math.round((maxEpochs / 5) * i), pad.left + (chartW / 5) * i, h - pad.bottom + 5);
        }
        // 轴名称
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText('训练轮数 (Epoch)', pad.left + chartW / 2, h - 2);
        ctx.save();
        ctx.translate(14, pad.top + chartH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('损失率 (Loss)', 0, 0);
        ctx.restore();

        if (lossHistory.length < 2) return;

        // 填充区域
        ctx.beginPath();
        ctx.moveTo(pad.left + chartW, pad.top + chartH);
        for (let i = lossHistory.length - 1; i >= 0; i--) {
            const x = pad.left + (i / maxEpochs) * chartW;
            const y = pad.top + (1 - lossHistory[i] / 5) * chartH;
            ctx.lineTo(x, y);
        }
        ctx.lineTo(pad.left, pad.top + chartH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartH);
        grad.addColorStop(0, 'rgba(239, 68, 68, 0.15)');
        grad.addColorStop(1, 'rgba(239, 68, 68, 0.01)');
        ctx.fillStyle = grad;
        ctx.fill();

        // 曲线
        ctx.beginPath();
        lossHistory.forEach((loss, i) => {
            const x = pad.left + (i / maxEpochs) * chartW;
            const y = pad.top + (1 - loss / 5) * chartH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // 曲线发光
        ctx.beginPath();
        lossHistory.forEach((loss, i) => {
            const x = pad.left + (i / maxEpochs) * chartW;
            const y = pad.top + (1 - loss / 5) * chartH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.lineWidth = 6;
        ctx.stroke();

        // 当前点
        if (lossHistory.length > 0) {
            const lastLoss = lossHistory[lossHistory.length - 1];
            const x = pad.left + ((lossHistory.length - 1) / maxEpochs) * chartW;
            const y = pad.top + (1 - lastLoss / 5) * chartH;
            ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#fff'; ctx.fill();
            ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)'; ctx.lineWidth = 2; ctx.stroke();
        }

        // 指标
        if (lossHistory.length > 0) {
            const lastLoss = lossHistory[lossHistory.length - 1];
            currentLossEl.textContent = lastLoss.toFixed(4);
            currentPplEl.textContent = Math.exp(lastLoss).toFixed(2);
            epochCountEl.textContent = lossHistory.length;
        }
    }

    function trainStep() {
        if (epochs >= maxEpochs) {
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            return;
        }
        const prevLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : 5.0;
        lossHistory.push(simulateLossStep(prevLoss));
        epochs++;
        const { w, h } = setupCanvas();
        drawLossChart(w, h);
        animId = requestAnimationFrame(trainStep);
    }

    function resetTraining() {
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        epochs = 0;
        lossHistory = [5.0];
        const { w, h } = setupCanvas();
        drawLossChart(w, h);
        setTimeout(trainStep, 300);
    }

    lrSlider.addEventListener('input', () => {
        lr = parseFloat(lrSlider.value);
        lrValue.textContent = lr.toFixed(3);
    });
    noiseSlider.addEventListener('input', () => {
        noise = parseFloat(noiseSlider.value);
        noiseValue.textContent = noise.toFixed(3);
    });
    resetBtn.addEventListener('click', resetTraining);

    setTimeout(resetTraining, 200);
    window.addEventListener('resize', () => {
        if (lossHistory.length > 0) { const { w, h } = setupCanvas(); drawLossChart(w, h); }
    });
}
