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
// ==================== 注意力积木 ====================
const ATTENTION_INFO = {
    flash:  { name: 'Flash Attention（默认）', desc: '内核融合实现，显存省、速度快；需要看注意力热力图时自动退回标准计算' },
    full:   { name: 'Full Attention', desc: '经典缩放点积注意力，全程显式矩阵运算，可用于可视化注意力权重' },
    linear: { name: 'Linear Attention', desc: '线性近似，复杂度 O(S)，超长序列友好；表达能力略降' },
};

async function initArchitectureOptions() {
    // 注意力积木下拉框（LLM / ViT / 多模态三处共用同一选项集）
    let names = ['flash', 'full', 'linear'];
    try {
        const res = await fetch('/api/architecture_options');
        const data = await res.json();
        if (data.status === 'success' && Array.isArray(data.attentions) && data.attentions.length) {
            names = data.attentions;
        }
    } catch (e) { /* 后端不可用时使用内置列表 */ }

    for (const selectId of ['llm_attn', 'vit_attn', 'mm_attn', 'cls_attn']) {
        const sel = document.getElementById(selectId);
        if (!sel) continue;
        sel.innerHTML = '';
        for (const n of names) {
            const opt = document.createElement('option');
            opt.value = n;
            opt.textContent = (ATTENTION_INFO[n] || {}).name || n;
            if (n === 'flash') opt.selected = true;
            sel.appendChild(opt);
        }
    }
    document.getElementById('llm_attn').addEventListener('change', () => updateAttentionDesc('llm'));
    updateAttentionDesc('llm');

    // 混合注意力搭建器：复用同一份可用注意力清单
    initAttnBuilder(names, 'llm');
    initAttnBuilder(names, 'vit');
}

function updateAttentionDesc(scope) {
    if (scope !== 'llm') return;
    const sel = document.getElementById('llm_attn');
    const desc = document.getElementById('llm_attn_desc');
    if (!sel || !desc) return;
    desc.textContent = (ATTENTION_INFO[sel.value] || {}).desc || '—';
}

// 图像分区状态：先选任务（分类/生成/编辑），分类任务下再选子架构（CNN/ViT）
let imageTask = 'cls';
let clsArch = 'cnn';

function currentImageModelKey() {
    if (imageTask === 'gen') return 'image_diffusion';
    if (imageTask === 'edit') return 'image_edit_diffusion';
    return clsArch === 'vit' ? 'image_vit' : 'image_cnn';
}

// 任务选择条：图像分类 / 图像生成 / 图像编辑（与顶部标签同款交互）
function switchImageTask(task) {
    imageTask = task;
    document.querySelectorAll('[data-img-task]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.imgTask === task);
    });
    // 二级架构条随任务切换（缩进在主条下方）
    document.getElementById('clsArchTabs').classList.toggle('hidden', task !== 'cls');
    document.getElementById('genArchTabs').classList.toggle('hidden', task !== 'gen');
    document.getElementById('editArchTabs').classList.toggle('hidden', task !== 'edit');
    syncImageTaskUI();
}

// 生成子类型：DDPM 标准（300步）/ DDIM 轻量快速（100步）
let genSub = 'ddpm';
function switchGenSub(sub) {
    genSub = sub;
    document.querySelectorAll('[data-gen-sub]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.genSub === sub);
    });
    if (sub === 'ddim') document.getElementById('diff_steps').value = 100;
    else document.getElementById('diff_steps').value = 300;
}

// 编辑子类型：标准（base=32）/ 轻量（base=16，CPU更快）
let editSub = 'std';
function switchEditSub(sub) {
    editSub = sub;
    document.querySelectorAll('[data-edit-sub]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.editSub === sub);
    });
    document.getElementById('diff_base').value = (sub === 'lite') ? 16 : 32;
}

// LLM 分支：文本生成(gen) / 语言分类(cls)
let llmMode = 'gen';
function switchLlmMode(mode) {
    llmMode = mode;
    document.querySelectorAll('[data-llm-task]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.llmTask === mode);
    });
    const isGen = mode === 'gen';
    document.getElementById('textParams').classList.toggle('hidden', !isGen);
    const clsCard = document.getElementById('textClsParams');
    if (clsCard) clsCard.classList.toggle('hidden', isGen);
    const tipG = document.getElementById('tipTextGen');
    const tipC = document.getElementById('tipTextCls');
    if (tipG) tipG.style.display = isGen ? 'block' : 'none';
    if (tipC) tipC.style.display = isGen ? 'none' : 'block';
}

// 分类子架构条：CNN / ViT
function switchClsArch(arch) {
    clsArch = arch;
    document.querySelectorAll('[data-cls-arch]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.clsArch === arch);
    });
    syncImageTaskUI();
}

function syncImageTaskUI() {
    const key = currentImageModelKey();
    const isCls = (key === 'image_cnn' || key === 'image_vit');
    document.getElementById('cnnParams').classList.toggle('hidden', key !== 'image_cnn');
    document.getElementById('vitParams').classList.toggle('hidden', key !== 'image_vit');
    document.getElementById('diffusionParams').classList.toggle(
        'hidden', !(key === 'image_diffusion' || key === 'image_edit_diffusion'));
    // 分类数量仅分类任务可见
    document.querySelectorAll('[data-cls-only]').forEach(el => {
        el.style.display = isCls ? '' : 'none';
    });
    // 扩散任务推荐小图尺寸
    const sizeInput = document.getElementById('image_size');
    if (key === 'image_diffusion' || key === 'image_edit_diffusion') {
        if (parseInt(sizeInput.value, 10) > 128) sizeInput.value = 64;
    }
}

function _normalizeSection(t) {
    // 兼容旧 localStorage 中的 'text'
    return t === 'llm' ? 'llm' : t;
}

function switchType(type) {
    currentType = type;

    // 切换标签样式
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === type);
    });

    // 参数面板显隐
    const show = id => document.getElementById(id).classList.remove('hidden');
    const hide = id => document.getElementById(id).classList.add('hidden');

    // 上传区与说明
    const tips = { llm: 'tipTextGen', image: 'imageUploadTip', multimodal: 'multimodalUploadTip' };
    Object.values(tips).forEach(id => document.getElementById(id).style.display = 'none');
    document.getElementById('imageUploadArea').style.display =
        (type === 'image' || type === 'multimodal') ? 'block' : 'none';
    document.getElementById('textUploadArea').style.display =
        type === 'llm' ? 'block' : 'none';
    document.getElementById(tips[type]).style.display = 'block';

    // 图像/LLM 任务条显隐；二级条先全隐再由各自切换函数恢复
    document.getElementById('imageTaskBar').classList.toggle('hidden', type !== 'image');
    document.getElementById('llmTaskBar').classList.toggle('hidden', type !== 'llm');
    ['clsArchTabs', 'genArchTabs', 'editArchTabs'].forEach(
        id => document.getElementById(id).classList.add('hidden'));
    if (type === 'image') switchImageTask(imageTask);
    if (type === 'llm') switchLlmMode(llmMode);

    // 图像任务条：仅图像分区显示；二级架构条全部先隐藏（由 syncImageTaskUI 决定）
    document.getElementById('imageTaskBar').classList.toggle('hidden', type !== 'image');
    ['clsArchTabs', 'genArchTabs', 'editArchTabs'].forEach(
        id => document.getElementById(id).classList.add('hidden'));
    if (type === 'image') {
        // 恢复当前任务对应的二级架构条与参数块
        switchImageTask(imageTask);
    }

    // 模型参数卡
    hide('imageParams'); hide('textParams'); hide('multimodalParams');
    const accMetric = document.getElementById('accuracyMetric');
    if (type === 'llm') {
        show('textParams');
        accMetric.style.display = 'none';
        document.getElementById('use_moe').disabled = false;
        document.getElementById('use_mla').disabled = false;
        if (document.getElementById('use_moe').checked) show('moeParams');
        if (document.getElementById('use_mla').checked) show('mlaParams');
    } else if (type === 'image') {
        show('imageParams');
        accMetric.style.display = 'flex';
        document.getElementById('use_moe').disabled = true;
        document.getElementById('use_mla').disabled = true;
        hide('moeParams'); hide('mlaParams');
        syncImageTaskUI();
    } else {
        show('multimodalParams');
        accMetric.style.display = 'flex';
        document.getElementById('use_moe').disabled = true;
        document.getElementById('use_mla').disabled = true;
        hide('moeParams'); hide('mlaParams');
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
    // 多模态分区复用图片上传区，但按 multimoal 类型入库（图文配对数据）
    formData.append('train_type', currentType === 'multimodal' ? 'multimodal' : 'image');

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
    const section = currentType;   // llm | image | multimodal
    const params = {
        train_type: section,
        dataset_id: selectedDatasetId,
        learning_rate: parseFloat(document.getElementById('learning_rate').value),
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batch_size').value),
    };

    if (section === 'llm') {
        if (llmMode === 'cls') {
            // 语言分类：字符级 Transformer 编码器
            params.model_key = 'text_classifier';
            params.vocab_size = parseInt(document.getElementById('cls_vocab_size').value);
            params.max_seq_len = parseInt(document.getElementById('cls_max_seq_len').value);
            params.d_model = parseInt(document.getElementById('cls_d_model').value);
            params.n_layers = parseInt(document.getElementById('cls_layers').value);
            params.n_heads = parseInt(document.getElementById('cls_heads').value);
            params.d_ff = parseInt(document.getElementById('cls_ff').value);
            params.attention_type = document.getElementById('cls_attn').value;
        } else {
            // 文本生成：Decoder-only Transformer + 积木式注意力 + MoE/MLA
            params.model_key = 'text_generation';
            params.vocab_size = parseInt(document.getElementById('vocab_size').value);
            params.max_seq_len = parseInt(document.getElementById('max_seq_len').value);
            params.d_model = parseInt(document.getElementById('d_model_text').value);
            params.n_layers = parseInt(document.getElementById('n_layers_text').value);
            params.n_heads = parseInt(document.getElementById('n_heads_text').value);
            params.d_ff = parseInt(document.getElementById('d_ff_text').value);
            params.dropout = parseFloat(document.getElementById('dropout_text').value);
            params.attention_type = document.getElementById('llm_attn').value;
            // 混合注意力搭建器：启用且已搭积木时携带逐层计划（后端校验并装配）
            const attnPlan = collectAttentionPlan('llm');
            if (attnPlan) params.attention_plan = attnPlan;
            params.use_moe = document.getElementById('use_moe').checked;
            params.use_mla = document.getElementById('use_mla').checked;
            if (params.use_moe) {
                params.moe_experts = parseInt(document.getElementById('moe_experts').value);
                params.moe_top_k = parseInt(document.getElementById('moe_top_k').value);
            }
            if (params.use_mla) {
                params.mla_dim = parseInt(document.getElementById('mla_dim').value);
            }
        }
        params.image_size = 224;
    } else if (section === 'image') {
        // 图像模型：任务条（分类/生成/编辑）+ 分类子架构条（CNN/ViT）积木式选择
        const arch = currentImageModelKey();
        params.model_key = arch;
        params.image_size = parseInt(document.getElementById('image_size').value);
        if (arch === 'image_cnn' || arch === 'image_vit') {
            params.num_classes = parseInt(document.getElementById('num_classes').value);
        }
        if (arch === 'image_cnn') {
            params.base_channels = parseInt(document.getElementById('base_channels').value);
        }
        if (arch === 'image_vit') {
            params.patch_size = parseInt(document.getElementById('patch_size').value);
            params.d_model = parseInt(document.getElementById('vit_d_model').value);
            params.n_layers = parseInt(document.getElementById('vit_layers').value);
            params.n_heads = parseInt(document.getElementById('vit_heads').value);
            params.d_ff = parseInt(document.getElementById('vit_ff').value);
            params.attention_type = document.getElementById('vit_attn').value;
            // ViT 混合注意力搭建器结果（未启用/未搭建时为 null）
            const vPlan = collectAttentionPlan('vit');
            if (vPlan) params.attention_plan = vPlan;
        }
        if (arch === 'image_diffusion' || arch === 'image_edit_diffusion') {
            params.base_channels = parseInt(document.getElementById('diff_base').value);
            params.num_timesteps = parseInt(document.getElementById('diff_steps').value);
        }
    } else {
        // 多模态单流：图文配对 + 积木式注意力
        params.model_key = 'multimodal_stream';
        params.vocab_size = parseInt(document.getElementById('mm_vocab_size').value);
        params.image_size = parseInt(document.getElementById('mm_image_size').value);
        params.patch_size = parseInt(document.getElementById('mm_patch_size').value);
        params.d_model = parseInt(document.getElementById('mm_d_model').value);
        params.n_layers = parseInt(document.getElementById('mm_layers').value);
        params.n_heads = parseInt(document.getElementById('mm_heads').value);
        params.d_ff = parseInt(document.getElementById('mm_ff').value);
        params.max_seq_len = parseInt(document.getElementById('mm_seq_len').value);
        params.attention_type = document.getElementById('mm_attn').value;
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
document.addEventListener('DOMContentLoaded', async () => {
    // 兼容旧版 localStorage 中遗留的 'text' 类型名
    if (currentType === 'text') {
        currentType = 'llm';
        localStorage.setItem('currentType', 'llm');
    }
    // 恢复训练类型
    if (currentType) switchType(currentType);

    // 加载注意力/架构积木选项
    await initArchitectureOptions();

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
        if (this.checked && currentType === 'llm') {
            document.getElementById('moeParams').classList.remove('hidden');
        } else {
            document.getElementById('moeParams').classList.add('hidden');
        }
    });

    document.getElementById('use_mla').addEventListener('change', function() {
        if (this.checked && currentType === 'llm') {
            document.getElementById('mlaParams').classList.remove('hidden');
        } else {
            document.getElementById('mlaParams').classList.add('hidden');
        }
    });
});

// ==================== 混合注意力积木搭建器（Scratch 式拖拽，多实例） ====================
// scope: 'llm'（文本生成）/ 'vit'（图像分类），各持独立序列状态与 DOM 元素组
const ATTN_BUILDER_SCOPES = {};   // scope -> { seq: [...], names: [...] }

function attnScopeIds(scope) {
    const p = scope === 'llm' ? '' : scope;
    return {
        toggle:   p ? p + 'UseAttnBuilder' : 'useAttnBuilder',
        box:      p ? p + 'AttnBuilder'    : 'attnBuilder',
        palette:  p ? p + 'AttnPalette'    : 'attnPalette',
        sequence: p ? p + 'AttnSequence'   : 'attnSequence',
        head:     p ? p + 'PlanHead'       : 'planHead',
        tail:     p ? p + 'PlanTail'       : 'planTail',
        preview:  p ? p + 'PlanPreview'    : 'attnPlanPreview',
        layers:   scope === 'llm' ? 'n_layers_text'
                  : scope === 'vit' ? 'vit_layers' : null,
    };
}

function attnChip(name, extraCls) {
    const label = (ATTENTION_INFO[name] || {}).name || name;
    // draggable 必须写在标签上：HTML5 拖放仅对 draggable=true 的元素触发 dragstart
    return '<span class="attn-block ' + (extraCls || '') +
           '" draggable="true" data-attn="' + name + '">' + label + '</span>';
}

function initAttnBuilder(names, scope) {
    const ids = attnScopeIds(scope);
    const palette = document.getElementById(ids.palette);
    if (!palette) return;
    ATTN_BUILDER_SCOPES[scope] = { seq: [], names: names.slice() };

    // 调色板：点击添加 / 拖入序列
    palette.innerHTML = names.map(n => attnChip(n)).join('');
    palette.querySelectorAll('.attn-block').forEach(el => {
        el.addEventListener('click', () => addToAttnSequence(scope, el.dataset.attn));
        el.addEventListener('dragstart', e =>
            e.dataTransfer.setData('text/plain', el.dataset.attn));
    });

    // 首尾特殊设置下拉（含"无"）
    const opts = ['<option value="">— 无 —</option>']
        .concat(names.map(n => '<option value="' + n + '">' +
            ((ATTENTION_INFO[n] || {}).name || n) + '</option>')).join('');
    const headSel = document.getElementById(ids.head);
    const tailSel = document.getElementById(ids.tail);
    headSel.innerHTML = opts;
    tailSel.innerHTML = opts;
    headSel.addEventListener('change', () => renderAttnPreview(scope));
    tailSel.addEventListener('change', () => renderAttnPreview(scope));

    // 序列容器拖放
    const zone = document.getElementById(ids.sequence);
    zone.addEventListener('dragover', e => {
        e.preventDefault(); zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag-over');
        const name = e.dataTransfer.getData('text/plain');
        if (name && names.includes(name)) addToAttnSequence(scope, name);
    });

    // 启用开关
    document.getElementById(ids.toggle).addEventListener('change', e => {
        document.getElementById(ids.box).classList.toggle('hidden', !e.target.checked);
        if (e.target.checked) renderAttnPreview(scope);
    });

    // 层数变化 → 刷新循环填充/截断预览
    if (ids.layers) {
        const layersEl = document.getElementById(ids.layers);
        if (layersEl) layersEl.addEventListener('input', () => {
            if (isAttnBuilderOn(scope)) renderAttnPreview(scope);
        });
    }

    // 清空按钮（作用域内 data-clear 标记）
    const clearBtn = document.getElementById(ids.box).querySelector('[data-clear]');
    if (clearBtn) clearBtn.addEventListener('click', () => clearAttnSequence(scope));

    renderAttnSequence(scope);
}

function isAttnBuilderOn(scope) {
    const cb = document.getElementById(attnScopeIds(scope).toggle);
    return !!(cb && cb.checked);
}

function addToAttnSequence(scope, name) {
    ATTN_BUILDER_SCOPES[scope].seq.push(name);
    renderAttnSequence(scope);
}

function removeAttnItem(scope, idx) {
    ATTN_BUILDER_SCOPES[scope].seq.splice(idx, 1);
    renderAttnSequence(scope);
}

function clearAttnSequence(scope) {
    ATTN_BUILDER_SCOPES[scope].seq = [];
    renderAttnSequence(scope);
}

function renderAttnSequence(scope) {
    const st = ATTN_BUILDER_SCOPES[scope];
    const zone = document.getElementById(attnScopeIds(scope).sequence);
    if (!st.seq.length) {
        zone.innerHTML = '<span class="attn-seq-empty">拖拽积木到这里，或点击上方调色板添加</span>';
        renderAttnPreview(scope);
        return;
    }
    zone.innerHTML = '';
    st.seq.forEach((name, i) => {
        const item = document.createElement('span');
        item.style.cssText = 'display:inline-flex;align-items:center;';
        item.innerHTML = attnChip(name, 'attn-seq-item') +
            '<span class="attn-seq-remove" title="移除">✕</span>' +
            (i < st.seq.length - 1
                ? '<span style="color:var(--text-muted);margin:0 2px;">→</span>'
                : '');
        const block = item.querySelector('.attn-block');
        block.addEventListener('dragstart', e =>
            e.dataTransfer.setData('application/x-attn-' + scope, String(i)));
        block.addEventListener('dragover', e => e.preventDefault());
        block.addEventListener('drop', e => {
            e.stopPropagation(); e.preventDefault();
            const from = parseInt(
                e.dataTransfer.getData('application/x-attn-' + scope), 10);
            if (!isNaN(from)) {
                const moved = st.seq.splice(from, 1)[0];
                st.seq.splice(i, 0, moved);
                renderAttnSequence(scope);
            }
        });
        item.querySelector('.attn-seq-remove').addEventListener(
            'click', () => removeAttnItem(scope, i));
        zone.appendChild(item);
    });
    renderAttnPreview(scope);
}

// 预览：与后端 resolve_attention_plan 同语义（循环填充/首尾覆盖/超限截断提醒）
function renderAttnPreview(scope) {
    const box = document.getElementById(attnScopeIds(scope).preview);
    if (!box) return;
    const st = ATTN_BUILDER_SCOPES[scope];
    if (!st || !st.seq.length) {
        box.textContent = '尚未搭建积木：将使用上方统一注意力。';
        return;
    }
    let nLayers = 4;
    const layersId = attnScopeIds(scope).layers;
    if (layersId) {
        const el = document.getElementById(layersId);
        if (el) nLayers = Math.max(1, parseInt(el.value || '4', 10));
    }
    const head = document.getElementById(attnScopeIds(scope).head).value;
    const tail = document.getElementById(attnScopeIds(scope).tail).value;

    let warn = '';
    if (st.seq.length > nLayers) {
        warn = '<span class="warn">⚠️ 搭建了 ' + st.seq.length +
               ' 块，超过模型层数 ' + nLayers +
               '，多余的将被自动截断</span><br>';
    }
    const plan = Array.from({ length: nLayers },
                            (_, i) => st.seq[i % st.seq.length]);
    if (head) plan[0] = head;
    if (tail) plan[nLayers - 1] = tail;

    box.innerHTML = warn + plan.map((a, i) =>
        '第' + String(i + 1).padStart(2, ' ') + '层 → ' + a +
        (head && i === 0 ? '  ⭐首层特殊' : '') +
        (tail && i === nLayers - 1 ? '  🌙尾层特殊' : '')).join('<br>');
}

// 收集搭建结果；未启用/未搭建返回 null（回退统一注意力）
function collectAttentionPlan(scope) {
    if (!isAttnBuilderOn(scope)) return null;
    const st = ATTN_BUILDER_SCOPES[scope];
    if (!st || !st.seq.length) return null;
    return {
        sequence: st.seq.slice(),
        head: document.getElementById(attnScopeIds(scope).head).value || null,
        tail: document.getElementById(attnScopeIds(scope).tail).value || null,
    };
}
