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
    initScrollAnimation();
    initNavHighlight();
    initSmoothScroll();
    initTokenDemo();
    initEmbeddingDemo();
    initAttentionDemo();
    initLossDemo();
    initGradientDemo();
    initTransformerTooltip();
    initActivationChart();
    initConvDemo();
    initTempDemo();
});

// ================================= 0. 粒子背景 =================================


// ================================= 滚动淡入动画 =================================
function initScrollAnimation() {
    const sections = document.querySelectorAll('section');
    if (!('IntersectionObserver' in window)) {
        // 环境不支持：直接全部显示（CSS 默认不隐藏，无需处理）
        return;
    }

    // 渐进增强：由 JS 打上 reveal-init 标记后才进入隐藏态，
    // 保证观察器失效时内容不会"永远透明"
    sections.forEach(sec => sec.classList.add('reveal-init'));

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);   // 已显示的不再观察
            }
        });
    }, { threshold: 0, rootMargin: '0px 0px -30px 0px' });
    sections.forEach(sec => observer.observe(sec));

    // 兜底：10 秒后强制全部显示，杜绝任何场景下的"内容不可见"
    setTimeout(() => {
        sections.forEach(sec => sec.classList.add('visible'));
    }, 10000);
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
    const words = text.match(/[a-zA-Z]+|\d+|\s+|[^\s]/g) || [text];
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

// ================================= 4. 梯度下降演示逻辑（二次函数） =================================
function initGradientDemo() {
    const canvas = document.getElementById('gradientCanvas');
    const ctx = canvas.getContext('2d');
    const lrSlider = document.getElementById('lrSlider');
    const lrValue = document.getElementById('lrValue');
    const resetBtn = document.getElementById('resetGradientBtn');

    // ---- 二次函数损失面：f(x) = 0.5·(x - x*)²，最优点 x*=0 ----
    const QUAD_A = 0.5;
    const X_MIN = 0;
    const X_VIEW = 10;            // 横轴显示范围 [-10, 10]
    const Y_VIEW_MAX = 44;        // 纵轴显示上限（f 最大约 40.5）

    let lr = 0.1;
    let ballX = 8;                // 参数起点
    let steps = 0;
    let diverged = false;
    let animationId = null;
    let trail = [];               // [{x, y}] 参数空间轨迹

    const fx = x => QUAD_A * (x - X_MIN) * (x - X_MIN);
    const dfx = x => 2 * QUAD_A * (x - X_MIN);

    function setupCanvas() {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.round(rect.width || 700);
        const h = Math.round(rect.height || 400);
        const pw = w * dpr, ph = h * dpr;
        // 同损失曲线：尺寸未变化时不重建画布位图
        if (canvas.width !== pw || canvas.height !== ph) {
            canvas.width = pw;
            canvas.height = ph;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        return { w, h };
    }

    // 参数空间 -> 像素坐标
    function toPX(x, w) {
        const padL = 46, padR = 16;
        return padL + ((x + X_VIEW) / (2 * X_VIEW)) * (w - padL - padR);
    }
    function toPY(y, h) {
        const padT = 16, padB = 30;
        return padT + (1 - y / Y_VIEW_MAX) * (h - padT - padB);
    }

    function drawAxesAndCurve(w, h) {
        // 坐标轴
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(toPX(X_MIN, w), toPY(0, h));
        ctx.lineTo(toPX(X_VIEW, w), toPY(0, h));   // x 轴
        ctx.moveTo(toPX(-X_VIEW, w), toPY(Y_VIEW_MAX, h));
        ctx.lineTo(toPX(-X_VIEW, w), toPY(0, h));  // y 轴
        ctx.stroke();

        // 刻度
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        for (let gx = -10; gx <= 10; gx += 5) {
            const px = toPX(gx, w);
            ctx.fillText(String(gx), px, toPY(0, h) + 16);
        }
        ctx.textAlign = 'right';
        ctx.fillText('f(x)', toPX(-X_VIEW, w) - 6, toPY(0, h) + 4);

        // 抛物线曲线
        ctx.beginPath();
        for (let px = toPX(-X_VIEW, w); px <= toPX(X_VIEW, w); px += 2) {
            const x = -X_VIEW + ((px - toPX(-X_VIEW, w)) /
                       (toPX(X_VIEW, w) - toPX(-X_VIEW, w))) * 2 * X_VIEW;
            const py = toPY(fx(x), h);
            if (px === toPX(-X_VIEW, w)) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = '#818cf8';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // 最优点标记（谷底）
        const minX = toPX(X_MIN, w), minY = toPY(0, h);
        ctx.beginPath();
        ctx.arc(minX, minY, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#22c55e';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(minX, minY, 11 + (Date.now() * 0.002) % 8, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.3)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('★ 最优点 x*=' + X_MIN, minX + 46, minY - 10);
    }

    function animateGradient() {
        const { w, h } = setupCanvas();

        // ===== 参数更新：x := x - lr·f'(x) =====
        const grad = dfx(ballX);
        ballX = ballX - lr * grad;
        steps++;

        // 发散保护：越界即钳制并标记
        if (!isFinite(ballX) || Math.abs(ballX) > X_VIEW) {
            ballX = Math.max(-X_VIEW, Math.min(X_VIEW, ballX));
            diverged = true;
        }

        trail.push({ x: ballX, y: fx(ballX) });
        if (trail.length > 200) trail.shift();

        // ===== 绘制帧 =====
        ctx.clearRect(0, 0, w, h);
        drawAxesAndCurve(w, h);

        // 轨迹点（沿曲线的衰减圆点）
        trail.forEach((p, i) => {
            ctx.beginPath();
            ctx.arc(toPX(p.x, w), toPY(p.y, h), 3.2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(239, 68, 68, ${0.15 + 0.85 * (i + 1) / trail.length})`;
            ctx.fill();
        });

        // 当前位置
        const bx = toPX(ballX, w), by = toPY(fx(ballX), h);
        const curGrad = dfx(ballX);

        // 切线段（展示该点的斜率方向）
        const tanLen = 34;
        const slopePx = -(curGrad) * (h - 46) / (Y_VIEW_MAX || 1) *
                        ((w - 62) / (2 * X_VIEW)) * -1;   // 像素斜率（y 轴向下）
        const dirX = 1, dirY = slopePx === 0 ? 0 : Math.sign(slopePx) *
                        Math.min(Math.abs(slopePx), 3);
        ctx.beginPath();
        ctx.moveTo(bx - tanLen, by - dirY * tanLen);
        ctx.lineTo(bx + tanLen, by + dirY * tanLen);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.fillStyle = '#f59e0b';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('切线斜率 = 梯度 ' + (curGrad >= 0 ? '+' : '') + curGrad.toFixed(2),
                     bx, by - 26);

        // 小球（含发光）
        ctx.beginPath();
        ctx.arc(bx, by, 14, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.18)';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(bx, by, 9, 0, Math.PI * 2);
        ctx.fillStyle = '#ef4444';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // 信息面板
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`f(x) = ${QUAD_A}(x−${X_MIN})²`, 12, 20);
        ctx.fillText(`当前参数 x = ${ballX.toFixed(3)}`, 12, 38);
        ctx.fillText(`损失 f(x) = ${fx(ballX).toFixed(4)}`, 12, 56);
        ctx.fillText(`梯度 f'(x) = ${curGrad.toFixed(3)} | 学习率 = ${lr}`, 12, 74);

        // 收敛 / 继续
        const converged = Math.abs(curGrad) < 0.005 && !diverged;
        if (converged) {
            ctx.fillStyle = '#22c55e';
            ctx.font = 'bold 14px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`✅ 已收敛到最优点！共 ${steps} 步`, w / 2, h - 36);
        } else if (diverged) {
            ctx.fillStyle = '#ef4444';
            ctx.font = 'bold 13px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('⚠️ 学习率过大导致发散（越过边界被钳制），点击重置调低学习率', w / 2, h - 36);
            return;  // 停止动画
        } else {
            animationId = requestAnimationFrame(animateGradient);
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
        ballX = 8;
        steps = 0;
        diverged = false;
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
            ctx.clearRect(0, 0, w, h);
            drawAxesAndCurve(w, h);
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
    let epochs = 0, maxEpochs = 500;
    let lossHistory = [5.0];   // 预置初始损失点：未开跑时坐标系与指标即有内容（epoch=0）
    let animId = null;
    let runToken = 0;          // 代际令牌：重置后旧动画循环自行退出，杜绝多循环并发
    let pendingTimer = null;

    function setupCanvas() {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.round(rect.width || 700), h = Math.round(rect.height || 350);
        const pw = w * dpr, ph = h * dpr;
        // 仅在像素尺寸真正变化时才重建画布位图（每帧重设是此前的性能热点）
        if (canvas.width !== pw || canvas.height !== ph) {
            canvas.width = pw;
            canvas.height = ph;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
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
        const yMax = computeYMax();   // 纵轴刻度按数据动态扩展

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
            ctx.fillText((yMax * (1 - i / 5)).toFixed(1), pad.left - 8, pad.top + (chartH / 5) * i);
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
            const y = pad.top + (1 - lossHistory[i] / yMax) * chartH;
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
            const y = pad.top + (1 - loss / yMax) * chartH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // 曲线发光
        ctx.beginPath();
        lossHistory.forEach((loss, i) => {
            const x = pad.left + (i / maxEpochs) * chartW;
            const y = pad.top + (1 - loss / yMax) * chartH;
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

    }

    // 指标每轮都刷新（与重绘节流解耦）；轮数用 epochs 计数器——
    // lossHistory 含预置的初始损失点，用其长度会多算 1 且未开跑就显示"1"
    function updateMetrics() {
        if (!lossHistory.length) return;
        const lastLoss = lossHistory[lossHistory.length - 1];
        currentLossEl.textContent = lastLoss.toFixed(4);
        currentPplEl.textContent = Math.exp(Math.min(lastLoss, 20)).toFixed(2);
        epochCountEl.textContent = String(epochs);
    }

    // 动态纵轴：高噪声下损失可能超过初始刻度顶值 5，
    // 按实际数据扩展刻度（含 15% 余量），避免曲线冲出绘图区
    function computeYMax() {
        return Math.max(5, ...lossHistory.map(v => v * 1.15));
    }

    function trainStep(token) {
        if (token !== runToken) return;   // 已被更新的重置取代：立即退出，防双循环叠加
        if (epochs >= maxEpochs) {
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            return;
        }
        const prevLoss = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : 5.0;
        lossHistory.push(simulateLossStep(prevLoss));
        epochs++;
        updateMetrics();   // 数字指标每轮都动（不随画布重绘节流）
        // 每 5 轮（含最后一轮）才做一次全量重绘：500 轮的重绘次数降为约 1/5
        if (epochs % 5 === 0 || epochs >= maxEpochs) {
            const { w, h } = setupCanvas();
            drawLossChart(w, h);
        }
        animId = requestAnimationFrame(() => trainStep(token));
    }

    function resetTraining() {
        runToken++;
        if (pendingTimer) { clearTimeout(pendingTimer); pendingTimer = null; }
        if (animId) { cancelAnimationFrame(animId); animId = null; }
        epochs = 0;
        lossHistory = [5.0];
        const { w, h } = setupCanvas();
        drawLossChart(w, h);
        updateMetrics();
        scheduleStart();   // 重置总是重新调度训练（不受懒启动一次性守卫限制）
    }

    // 首次滚入视口才开始训练动画：
    // 页面一打开就在后台跑满 500 轮，用户滚动到此处时只剩静态成品
    function scheduleStart() {
        const token = runToken;
        pendingTimer = setTimeout(() => {
            pendingTimer = null;
            trainStep(token);
        }, 300);
    }
    let started = false;
    function startOnce() {
        if (started) return;   // 仅约束"滚入视口自动触发"，不拦重置按钮
        started = true;
        scheduleStart();
    }

    const fmtLr = v => (v >= 0.001 ? String(+v.toFixed(6)) : v.toExponential(0));
    lrSlider.addEventListener('input', () => {
        lr = parseFloat(lrSlider.value);
        lrValue.textContent = fmtLr(lr);
    });
    noiseSlider.addEventListener('input', () => {
        noise = parseFloat(noiseSlider.value);
        noiseValue.textContent = noise.toFixed(3);
    });
    resetBtn.addEventListener('click', resetTraining);

    // 初始仅绘制坐标系与起始点；首次滚入视口才开始演示
    (() => {
        const { w, h } = setupCanvas();
        drawLossChart(w, h);
        updateMetrics();
    })();
    const lossSection = canvas.closest('section');
    if ('IntersectionObserver' in window && lossSection) {
        const io = new IntersectionObserver((entries) => {
            entries.forEach(en => {
                if (en.isIntersecting) {
                    io.disconnect();
                    startOnce();
                }
            });
        }, { threshold: 0 });
        io.observe(lossSection);
    } else {
        startOnce();
    }
    window.addEventListener('resize', () => {
        if (lossHistory.length > 0) { const { w, h } = setupCanvas(); drawLossChart(w, h); }
    });
}

// ================================= 6. 激活函数曲线对比（静态绘制） =================================
function initActivationChart() {
    const canvas = document.getElementById('actCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { top: 20, bottom: 30, left: 45, right: 15 };
    const xMin = -5, xMax = 5, yMin = -1.2, yMax = 3;

    const sigmoid = x => 1 / (1 + Math.exp(-x));
    const tanh = x => Math.tanh(x);
    const relu = x => Math.max(0, x);
    const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));

    function toX(x) { return pad.left + ((x - xMin) / (xMax - xMin)) * (W - pad.left - pad.right); }
    function toY(y) { return pad.top + (1 - (y - yMin) / (yMax - yMin)) * (H - pad.top - pad.bottom); }

    ctx.clearRect(0, 0, W, H);
    // 网格与坐标轴
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '11px Inter, sans-serif';
    ctx.lineWidth = 1;
    for (let gx = xMin; gx <= xMax; gx++) {
        ctx.beginPath(); ctx.moveTo(toX(gx), pad.top); ctx.lineTo(toX(gx), H - pad.bottom); ctx.stroke();
        if (gx % 2 === 0) ctx.fillText(gx, toX(gx) - 6, H - pad.bottom + 16);
    }
    for (let gy = yMin; gy <= yMax; gy++) {
        ctx.beginPath(); ctx.moveTo(pad.left, toY(gy)); ctx.lineTo(W - pad.right, toY(gy)); ctx.stroke();
        if (Number.isInteger(gy)) ctx.fillText(gy, pad.left - 22, toY(gy) + 4);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.beginPath(); ctx.moveTo(pad.left, toY(0)); ctx.lineTo(W - pad.right, toY(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(toX(0), pad.top); ctx.lineTo(toX(0), H - pad.bottom); ctx.stroke();

    function plot(fn, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        let started = false;
        for (let px = pad.left; px <= W - pad.right; px += 2) {
            const x = xMin + ((px - pad.left) / (W - pad.left - pad.right)) * (xMax - xMin);
            const y = fn(x);
            if (!isFinite(y)) { started = false; continue; }
            const py = Math.max(pad.top - 40, Math.min(H - pad.bottom + 40, toY(y)));
            if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    plot(sigmoid, '#ef4444');
    plot(tanh, '#f59e0b');
    plot(relu, '#22c55e');
    plot(gelu, '#818cf8');
}

// ================================= 7. 卷积扫描动画 =================================
function initConvDemo() {
    const gridEl = document.getElementById('convGrid');
    const outEl = document.getElementById('convOut');
    if (!gridEl || !outEl) return;

    const IN = 6, OUT = 4, CELL = 34;
    // 固定的"边缘检测"式权重，让输出有可解释的高低响应
    const kernel = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
    const input = [];
    let seed = 42;
    const rand = () => (seed = (seed * 9301 + 49297) % 233280) / 233280;
    for (let r = 0; r < IN; r++) {
        input.push([]);
        for (let c = 0; c < IN; c++) input[r].push(rand());
    }

    gridEl.style.gridTemplateColumns = `repeat(${IN}, ${CELL}px)`;
    outEl.style.gridTemplateColumns = `repeat(${OUT}, ${CELL}px)`;
    const cells = [], outCells = [];
    for (let r = 0; r < IN; r++) for (let c = 0; c < IN; c++) {
        const d = document.createElement('div');
        d.className = 'conv-cell';
        d.style.opacity = 0.25 + input[r][c] * 0.75;
        gridEl.appendChild(d); cells.push(d);
    }
    for (let i = 0; i < OUT * OUT; i++) {
        const d = document.createElement('div');
        d.className = 'conv-cell out-cell';
        outEl.appendChild(d); outCells.push(d);
    }

    let step = 0;
    function compute(or_, oc) {
        let s = 0;
        for (let kr = 0; kr < 3; kr++) for (let kc = 0; kc < 3; kc++)
            s += input[or_ + kr][oc + kc] * kernel[kr][kc];
        return s;
    }
    function tick() {
        cells.forEach(c => c.classList.remove('active'));
        if (step >= OUT * OUT) {
            step = 0;
            outCells.forEach(c => { c.classList.remove('hot'); c.textContent = ''; });
        } else {
            const or_ = Math.floor(step / OUT), oc = step % OUT;
            for (let kr = 0; kr < 3; kr++) for (let kc = 0; kc < 3; kc++)
                cells[(or_ + kr) * IN + (oc + kc)].classList.add('active');
            const val = compute(or_, oc);
            const cell = outCells[step];
            cell.classList.add('hot');
            cell.style.setProperty('--heat', Math.min(1, Math.abs(val) / 2).toFixed(2));
            cell.textContent = (val >= 0 ? '+' : '') + val.toFixed(1);
            step++;
        }
        setTimeout(tick, 700);
    }
    tick();
}

// ================================= 8. 温度采样概率条 =================================
function initTempDemo() {
    const slider = document.getElementById('tempSlider');
    const valueEl = document.getElementById('tempValue');
    const barsEl = document.getElementById('probBars');
    if (!slider || !barsEl) return;

    const candidates = [
        { ch: '天', logit: 3.0 }, { ch: '很', logit: 2.5 },
        { ch: '气', logit: 2.0 }, { ch: '香', logit: 1.0 },
    ];

    function softmaxT(logits, T) {
        const scaled = logits.map(l => l / T);
        const m = Math.max(...scaled);
        const exps = scaled.map(s => Math.exp(s - m));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    function render(T) {
        valueEl.textContent = T.toFixed(1);
        const probs = softmaxT(candidates.map(c => c.logit), T);
        barsEl.innerHTML = '';
        candidates.forEach((c, i) => {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML =
                `<span class="prob-label">「${c.ch}」</span>` +
                `<div class="prob-track"><div class="prob-fill" style="width:${(probs[i] * 100).toFixed(1)}%"></div></div>` +
                `<span class="prob-pct">${(probs[i] * 100).toFixed(1)}%</span>`;
            barsEl.appendChild(row);
        });
    }

    slider.addEventListener('input', () => render(parseFloat(slider.value)));
    render(0.8);
}
