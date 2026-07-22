// ==================== 全局状态 ====================
let currentType = localStorage.getItem('currentType') || 'image';
let currentTaskId = localStorage.getItem('currentTaskId') || null;
let statusInterval = null;
let uploadedFileCount = 0;
let selectedDatasetId = null;
let selectedClassFile = null;
let classPreviewData = null;
let selectedTextFile = null;

// ==================== 训练类型切换 ====================
function switchType(type) {
    currentType = type;
    
    // 切换标签样式
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === type);
    });
    
    // 切换参数面板
    const imgPanel = document.getElementById('imageParams');
    const txtPanel = document.getElementById('textParams');
    const accMetric = document.getElementById('accuracyMetric');
    
    if (type === 'image') {
        imgPanel.classList.remove('hidden');
        txtPanel.classList.add('hidden');
        accMetric.style.display = 'flex';
        // 切换上传区
        document.getElementById('imageUploadTip').style.display = 'block';
        document.getElementById('textUploadTip').style.display = 'none';
        document.getElementById('imageUploadArea').style.display = 'block';
        document.getElementById('textUploadArea').style.display = 'none';
        // 图片训练禁用MoE/MLA
        document.getElementById('use_moe').disabled = true;
        document.getElementById('use_mla').disabled = true;
        document.getElementById('moeParams').classList.add('hidden');
        document.getElementById('mlaParams').classList.add('hidden');
    } else {
        imgPanel.classList.add('hidden');
        txtPanel.classList.remove('hidden');
        accMetric.style.display = 'none';
        // 切换上传区
        document.getElementById('imageUploadTip').style.display = 'none';
        document.getElementById('textUploadTip').style.display = 'block';
        document.getElementById('imageUploadArea').style.display = 'none';
        document.getElementById('textUploadArea').style.display = 'block';
        // 文本训练启用MoE/MLA，根据勾选状态显示参数
        document.getElementById('use_moe').disabled = false;
        document.getElementById('use_mla').disabled = false;
        if (document.getElementById('use_moe').checked) {
            document.getElementById('moeParams').classList.remove('hidden');
        }
        if (document.getElementById('use_mla').checked) {
            document.getElementById('mlaParams').classList.remove('hidden');
        }
    }
}

// ==================== 图片分类数据集上传逻辑 ====================
// 绑定上传区点击
document.getElementById('classUploadArea').addEventListener('click', () => {
    document.getElementById('classFileInput').click();
});

// 绑定拖拽
const classUploadArea = document.getElementById('classUploadArea');
classUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    classUploadArea.classList.add('drag-over');
});
classUploadArea.addEventListener('dragleave', () => {
    classUploadArea.classList.remove('drag-over');
});
classUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    classUploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].name.endsWith('.zip')) {
        handleClassFileSelect(files[0]);
    } else {
        showClassError('请上传 .zip 格式的文件');
    }
});

// 绑定文件选择
document.getElementById('classFileInput').addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleClassFileSelect(e.target.files[0]);
    }
});

// 处理ZIP选择
async function handleClassFileSelect(file) {
    if (!file.name.endsWith('.zip')) {
        showClassError('请上传 .zip 格式的文件');
        return;
    }

    selectedClassFile = file;
    showClassError('');
    document.getElementById('classUploadStatus').innerHTML = `
        <div class="upload-progress">⏳ 正在解析 ZIP 结构...</div>
    `;

    const formData = new FormData();
    formData.append('zip_file', file);

    try {
        const res = await fetch('/api/preview_zip', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.status === 'success') {
            classPreviewData = data;
            renderClassPreview(data);
        } else {
            showClassError(data.message || 'ZIP 解析失败');
            document.getElementById('classUploadStatus').innerHTML = '';
        }
    } catch (err) {
        showClassError('解析失败: ' + err.message);
        document.getElementById('classUploadStatus').innerHTML = '';
    }
}

// 渲染ZIP预览
function renderClassPreview(data) {
    const previewEl = document.getElementById('classZipPreview');
    const classGrid = document.getElementById('classGrid');
    const totalClass = document.getElementById('classTotalClass');
    const confirmBtn = document.getElementById('confirmBtn');

    classGrid.innerHTML = data.classes.map(cls => `
        <div class="class-card">
            <div class="class-name">
                📁 ${cls.name}
                <span class="class-count">${cls.count} 张</span>
            </div>
            <div class="class-images">
                ${cls.samples.map(img => `
                    <img src="${img}" alt="示例">
                `).join('')}
            </div>
        </div>
    `).join('');

    totalClass.textContent = `${data.total_classes} 个类别，共 ${data.total_images} 张图片`;
    previewEl.classList.add('show');
    confirmBtn.classList.add('show');
    document.getElementById('classUploadStatus').innerHTML = '';
}

// 确认图片上传
async function confirmClassUpload() {
    if (!selectedClassFile || !classPreviewData) return;

    const btn = document.getElementById('confirmBtn');
    btn.disabled = true;
    btn.textContent = '⏳ 上传中...';

    const formData = new FormData();
    formData.append('file', selectedClassFile);
    formData.append('train_type', 'image');

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const contentType = res.headers.get('Content-Type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('后端返回异常，请检查服务是否正常');
        }
        
        const data = await res.json();

        if (data.status === 'success') {
            uploadedFileCount = data.file_count || 0;
            document.getElementById('fileCountBadge').textContent = `${uploadedFileCount} 张图片`;
            document.getElementById('classUploadStatus').innerHTML = `
                <div class="upload-success">
                    ✅ ${data.message}
                </div>
            `;
            if (data.file_id) {
                selectedDatasetId = data.file_id;
            }
            loadDatasets();
            resetUploadArea();
        } else {
            showClassError(data.message || '上传失败');
            btn.disabled = false;
            btn.textContent = '✅ 确认上传数据集';
        }
    } catch (err) {
        showClassError('上传失败: ' + err.message);
        btn.disabled = false;
        btn.textContent = '✅ 确认上传数据集';
    }
}

// 图片上传错误提示
function showClassError(msg) {
    const errorBox = document.getElementById('classErrorBox');
    if (msg) {
        errorBox.textContent = '❌ ' + msg;
        errorBox.classList.add('show');
    } else {
        errorBox.classList.remove('show');
    }
}

// ==================== 文本训练上传逻辑 ====================
// 绑定文本上传区点击
document.getElementById('textUploadZone').addEventListener('click', () => {
    document.getElementById('textFileInput').click();
});

// 绑定文本拖拽
const textUploadZone = document.getElementById('textUploadZone');
textUploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    textUploadZone.classList.add('drag-over');
});
textUploadZone.addEventListener('dragleave', () => {
    textUploadZone.classList.remove('drag-over');
});
textUploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    textUploadZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.name.endsWith('.txt') || file.name.endsWith('.zip')) {
            handleTextFileSelect(file);
        } else {
            showTextError('请上传 .txt 或 .zip 格式的文件');
        }
    }
});

// 绑定文本文件选择
document.getElementById('textFileInput').addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleTextFileSelect(e.target.files[0]);
    }
});

// 处理文本文件选择
async function handleTextFileSelect(file) {
    if (!file.name.endsWith('.txt') && !file.name.endsWith('.zip')) {
        showTextError('请上传 .txt 或 .zip 格式的文件');
        return;
    }

    selectedTextFile = file;
    showTextError('');
    document.getElementById('textUploadStatus').innerHTML = `
        <div class="upload-progress">⏳ 正在上传文本数据...</div>
    `;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('train_type', 'text');

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const contentType = res.headers.get('Content-Type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('后端返回异常，请检查服务是否正常');
        }
        
        const data = await res.json();

        if (data.status === 'success') {
            document.getElementById('textUploadStatus').innerHTML = `
                <div class="upload-success">
                    ✅ ${data.message}
                </div>
            `;
            if (data.file_id) {
                selectedDatasetId = data.file_id;
            }
            loadDatasets();
            selectedTextFile = null;
            document.getElementById('textFileInput').value = '';
        } else {
            showTextError(data.message || '上传失败');
        }
    } catch (err) {
        showTextError('上传失败: ' + err.message);
    }
}

// 文本上传错误提示
function showTextError(msg) {
    const errorBox = document.getElementById('textErrorBox');
    if (msg) {
        errorBox.textContent = '❌ ' + msg;
        errorBox.classList.add('show');
    } else {
        errorBox.classList.remove('show');
    }
}

// ==================== 数据集选择逻辑 ====================
// 加载数据集列表
async function loadDatasets() {
    try {
        const res = await fetch('/api/list_datasets');
        const data = await res.json();
        const container = document.getElementById('datasetList');
        const countBadge = document.getElementById('datasetCount');
        
        if (!data.datasets || data.datasets.length === 0) {
            container.innerHTML = '<p class="empty-state">暂无数据集，请先上传训练数据</p>';
            countBadge.textContent = '0 个数据集';
            selectedDatasetId = null;
            return;
        }
        
        countBadge.textContent = `${data.datasets.length} 个数据集`;
        
        // 默认选中第一个（最新的）
        if (!selectedDatasetId || !data.datasets.find(d => d.id === selectedDatasetId)) {
            selectedDatasetId = data.datasets[0].id;
        }
        
        container.innerHTML = data.datasets.map(ds => {
            const typeIcon = ds.type === 'image' ? '🖼️' : '📝';
            const typeLabel = ds.type === 'image' ? '图片' : '文本';
            return `
                <div class="dataset-item ${ds.id === selectedDatasetId ? 'selected' : ''}" 
                     onclick="selectDataset(${ds.id})">
                    <div class="dataset-info">
                        <span class="dataset-icon">${typeIcon}</span>
                        <div class="dataset-details">
                            <span class="dataset-name" title="${ds.name}">${ds.name}</span>
                            <div class="dataset-meta">
                                <span class="dataset-count">${ds.sample_count} 个样本</span>
                                <span class="dataset-type">${typeLabel}</span>
                                <span class="dataset-time">${ds.created_at.split('T')[0]}</span>
                            </div>
                        </div>
                    </div>
                    <div class="dataset-check">
                        ${ds.id === selectedDatasetId ? '✅ 已选中' : ''}
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error('[Datasets] 加载失败:', err);
    }
}

// 切换选中数据集
function selectDataset(datasetId) {
    selectedDatasetId = datasetId;
    loadDatasets();
}

// 重置上传区域
function resetUploadArea() {
    if (currentType === 'image') {
        selectedClassFile = null;
        classPreviewData = null;
        document.getElementById('classFileInput').value = '';
        document.getElementById('classZipPreview').classList.remove('show');
        document.getElementById('confirmBtn').classList.remove('show');
        document.getElementById('classErrorBox').classList.remove('show');
        document.getElementById('classUploadStatus').innerHTML = '';
    } else {
        selectedTextFile = null;
        document.getElementById('textFileInput').value = '';
        document.getElementById('textErrorBox').classList.remove('show');
        document.getElementById('textUploadStatus').innerHTML = '';
    }
}

// ==================== 训练相关逻辑 ====================
// 开始训练
async function startTraining() {
    const btn = document.getElementById('startBtn');
    btn.disabled = true;
    btn.textContent = '⏳ 启动中...';
    
    // 校验数据集
    if (!selectedDatasetId) {
        alert('请先选择训练数据集');
        btn.disabled = false;
        btn.textContent = '🚀 开始训练';
        return;
    }

    // 校验数据集类型匹配
    try {
        const res = await fetch('/api/list_datasets');
        const data = await res.json();
        const selectedDs = data.datasets.find(d => d.id === selectedDatasetId);
        if (selectedDs && selectedDs.type !== currentType) {
            alert(`当前选中的是${selectedDs.type === 'image' ? '图片' : '文本'}数据集，请切换到${currentType === 'image' ? '图片' : '文本'}训练类型，或选择对应类型的数据集`);
            btn.disabled = false;
            btn.textContent = '🚀 开始训练';
            return;
        }
    } catch (e) {
        console.error('数据集校验失败:', e);
    }
    
    // 收集参数
    const params = {
        train_type: currentType,
        dataset_id: selectedDatasetId,
        learning_rate: parseFloat(document.getElementById('learning_rate').value),
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
    };
    
    if (currentType === 'image') {
        params.image_size = parseInt(document.getElementById('image_size').value);
        params.num_classes = parseInt(document.getElementById('num_classes').value);
        params.base_channels = parseInt(document.getElementById('base_channels').value);
        params.d_model = 512; params.n_layers = 6; params.n_heads = 8;
        params.d_ff = 2048; params.dropout = 0.1;
        params.use_moe = false;
        params.use_mla = false;
        params.vocab_size = 1000; params.max_seq_len = 128;
    } else {
        params.vocab_size = parseInt(document.getElementById('vocab_size').value);
        params.max_seq_len = parseInt(document.getElementById('max_seq_len').value);
        params.d_model = parseInt(document.getElementById('d_model_text').value);
        params.n_layers = parseInt(document.getElementById('n_layers_text').value);
        params.n_heads = parseInt(document.getElementById('n_heads_text').value);
        params.d_ff = parseInt(document.getElementById('d_ff_text').value);
        params.dropout = parseFloat(document.getElementById('dropout_text').value);
        params.use_moe = document.getElementById('use_moe').checked;
        params.use_mla = document.getElementById('use_mla').checked;
        // 收集MoE/MLA专属参数
        if (params.use_moe) {
            params.moe_experts = parseInt(document.getElementById('moe_experts').value);
            params.moe_top_k = parseInt(document.getElementById('moe_top_k').value);
        }
        if (params.use_mla) {
            params.mla_heads = parseInt(document.getElementById('mla_heads').value);
            params.mla_dim = parseInt(document.getElementById('mla_dim').value);
        }
        params.image_size = 224; params.num_classes = 10; params.base_channels = 64;
    }
    
    // 重置状态
    document.getElementById('statusBadge').textContent = '🔄 启动中...';
    document.getElementById('statusBadge').style.background = 'var(--accent)';
    document.getElementById('statusMessage').textContent = '正在连接训练服务...';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('lossValue').textContent = '—';
    document.getElementById('accValue').textContent = '—';
    document.getElementById('progressPercent').textContent = '0%';
    
    // 超时处理
    const timeoutId = setTimeout(() => {
        if (btn.disabled) {
            btn.disabled = false;
            btn.textContent = '🚀 开始训练';
            document.getElementById('statusMessage').textContent = '⚠️ 启动超时，请检查后端服务';
        }
    }, 30000);
    
    try {
        const res = await fetch('/api/start_training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const data = await res.json();
        clearTimeout(timeoutId);
        
        if (data.status === 'success') {
            currentTaskId = data.task_id;
            localStorage.setItem('currentTaskId', currentTaskId);
            document.getElementById('statusBadge').textContent = '🔄 运行中';
            document.getElementById('statusMessage').textContent = data.message;
            if (statusInterval) clearInterval(statusInterval);
            statusInterval = setInterval(pollStatus, 500);
        } else {
            btn.disabled = false;
            btn.textContent = '🚀 开始训练';
            document.getElementById('statusMessage').textContent = '❌ ' + data.message;
        }
    } catch (err) {
        clearTimeout(timeoutId);
        btn.disabled = false;
        btn.textContent = '🚀 开始训练';
        document.getElementById('statusMessage').textContent = '❌ 请求失败: ' + err.message;
    }
}

// 轮询训练状态
async function pollStatus() {
    if (!currentTaskId) {
        clearInterval(statusInterval);
        statusInterval = null;
        return;
    }
    
    try {
        const res = await fetch(`/api/task_status/${currentTaskId}`);
        const data = await res.json();
        
        if (data.status === 'not_found') {
            clearInterval(statusInterval);
            statusInterval = null;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('startBtn').textContent = '🚀 开始训练';
            document.getElementById('statusMessage').textContent = '⚠️ 任务已失效，请重新开始';
            return;
        }
        
        // 更新进度
        const progress = data.progress || 0;
        document.getElementById('progressFill').style.width = progress + '%';
        document.getElementById('progressPercent').textContent = Math.round(progress) + '%';
        document.getElementById('statusMessage').textContent = data.message || '训练中...';
        
        // 更新Loss
        if (data.loss !== null && data.loss !== undefined) {
            document.getElementById('lossValue').textContent = typeof data.loss === 'number' ? data.loss.toFixed(4) : data.loss;
        }
        
        // 更新准确率（仅图片训练显示）
        const accEl = document.getElementById('accValue');
        const accMetric = document.getElementById('accuracyMetric');
        if (data.accuracy !== null && data.accuracy !== undefined && currentType === 'image') {
            accEl.textContent = typeof data.accuracy === 'number' ? data.accuracy.toFixed(2) : data.accuracy;
            accMetric.style.display = 'flex';
        } else {
            accMetric.style.display = 'none';
        }
        
        // 更新状态徽章
        const badge = document.getElementById('statusBadge');
        if (data.status === 'running') {
            badge.textContent = '🔄 运行中';
            badge.style.background = 'var(--accent)';
        } else if (data.status === 'completed') {
            badge.textContent = '✅ 已完成';
            badge.style.background = '#22c55e';
            clearInterval(statusInterval);
            statusInterval = null;
            localStorage.removeItem('currentTaskId');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('startBtn').textContent = '🚀 开始训练';
            setTimeout(loadModels, 1000);
        } else if (data.status === 'failed') {
            badge.textContent = '❌ 失败';
            badge.style.background = '#ef4444';
            clearInterval(statusInterval);
            statusInterval = null;
            localStorage.removeItem('currentTaskId');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('startBtn').textContent = '🚀 开始训练';
        }
    } catch (err) {
        console.error('[Poll] 请求失败:', err);
    }
}

// ==================== 模型管理逻辑 ====================
// 加载模型列表
async function loadModels() {
    try {
        const res = await fetch('/api/list_models');
        const data = await res.json();
        const container = document.getElementById('modelsContainer');
        const countBadge = document.getElementById('modelCount');
        
        if (!data.models || data.models.length === 0) {
            container.innerHTML = '<p class="empty-state">训练完成后，模型将出现在这里</p>';
            countBadge.textContent = '0';
            return;
        }
        
        countBadge.textContent = data.models.length;
        container.innerHTML = data.models.map(m => {
            let icon = '📦';
            if (m.name.includes('cnn')) icon = '🖼️';
            else if (m.name.includes('text')) icon = '📝';
            return `
                <div class="model-item">
                    <div class="model-info">
                        <span class="model-icon">${icon}</span>
                        <div class="model-details">
                            <span class="model-name">${m.name}</span>
                            <div class="model-meta">
                                <span class="model-size">${m.size}</span>
                            </div>
                        </div>
                    </div>
                    <button class="btn-download" onclick="downloadModel('${m.name}')">⬇️ 下载</button>
                </div>
            `;
        }).join('');
    } catch (err) {
        console.error('[Models] 加载失败:', err);
    }
}

// 下载模型
async function downloadModel(filename) {
    try {
        const res = await fetch(`/api/download_model/${filename}`);
        if (!res.ok) {
            alert('下载失败');
            return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (err) {
        alert('下载失败: ' + err.message);
    }
}

// ==================== 退出登录 ====================
async function handleLogout() {
    if (!confirm('确定要退出登录吗？')) {
        return;
    }
    
    try {
        const res = await fetch('/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            // 清除前端状态
            currentTaskId = null;
            if (statusInterval) {
                clearInterval(statusInterval);
                statusInterval = null;
            }
            selectedDatasetId = null;
            selectedClassFile = null;
            classPreviewData = null;
            selectedTextFile = null;
            
            // 跳转登录页
            window.location.href = '/login';
        } else {
            alert('退出失败: ' + data.message);
        }
    } catch (err) {
        alert('退出失败: ' + err.message);
    }
}

// ==================== 页面初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    // 恢复训练类型
    if (currentType) switchType(currentType);
    
    // 加载模型和数据集列表
    loadModels();
    loadDatasets();

    // 恢复之前未完成的训练任务
    if (currentTaskId) {
        pollStatus();
        statusInterval = setInterval(pollStatus, 500);
        document.getElementById('statusBadge').textContent = '🔄 恢复中...';
        document.getElementById('statusMessage').textContent = '正在恢复之前的训练任务...';
    }

    // MoE/MLA复选框联动参数显示
    document.getElementById('use_moe').addEventListener('change', function() {
        if (this.checked && currentType === 'text') {
            document.getElementById('moeParams').classList.remove('hidden');
        } else {
            document.getElementById('moeParams').classList.add('hidden');
        }
    });

    document.getElementById('use_mla').addEventListener('change', function() {
        if (this.checked && currentType === 'text') {
            document.getElementById('mlaParams').classList.remove('hidden');
        } else {
            document.getElementById('mlaParams').classList.add('hidden');
        }
    });
});