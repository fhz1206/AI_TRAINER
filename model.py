import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image as PILImage
from os import walk, path as os_path
from os.path import basename, dirname, join as path_join, exists as path_exists
from random import randint, seed as random_seed
from time import time as time_now
from collections import OrderedDict
from threading import Lock

# -------------------------- 全局配置 --------------------------
warnings.filterwarnings('ignore')
random_seed(42)  # 固定随机种子保证可复现

# -------------------------- 工具组件：线程安全LRU缓存 --------------------------
class LRUCache:
    """线程安全的LRU缓存，用于缓存已加载的图像减少磁盘IO"""
    def __init__(self, capacity=500):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        with self.lock:
            self.cache.clear()

# -------------------------- 数据集类 --------------------------
class ImageDataset(Dataset):
    """图像分类数据集，支持任意层级目录结构，自动推断类别，自带LRU缓存"""
    def __init__(self, folder_path, image_size=224, cache_capacity=500,
                 progress_callback=None, scan_interval=100):
        self.folder_path = folder_path
        self.image_size = image_size
        self.scan_interval = scan_interval
        self.cache = LRUCache(capacity=cache_capacity)
        self.image_files = []

        # 扫描图像文件
        print(f"[ImageDataset] 开始扫描: {folder_path}")
        if path_exists(folder_path):
            self._scan_with_progress(folder_path, progress_callback)
        print(f"[ImageDataset] 共 {len(self.image_files)} 个文件")

        # 自动推断类别：直接父文件夹名为类别，根目录下的图片归为「默认」类
        self.classes = {}
        for img_path in self.image_files:
            parent_dir = os_path.dirname(img_path)
            class_name = basename(parent_dir)
            if os_path.relpath(parent_dir, self.folder_path) == '.':
                class_name = '默认'
            if class_name and class_name not in self.classes:
                self.classes[class_name] = len(self.classes)
        self.num_classes = max(len(self.classes), 1)
        print(f"[ImageDataset] 类别: {self.classes}，共 {self.num_classes} 类")

        # ImageNet标准图像预处理
        self.transform = Compose([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 缓存统计
        self.access_count = 0
        self.cache_hit_count = 0

    def _scan_with_progress(self, folder_path, callback):
        """带进度回调的目录扫描"""
        count = 0
        start = time_now()
        for root, _, files in walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_files.append(path_join(root, f))
                    count += 1
                    if count % self.scan_interval == 0 and callback:
                        callback(count, '?', f'扫描中... 已发现 {count} 个文件 ({time_now()-start:.0f}s)')
        if callback:
            callback(count, count, f'扫描完成，共 {count} 个文件 ({time_now()-start:.0f}s)')

    def __len__(self):
        return max(len(self.image_files), 1)

    def _load_image(self, img_path):
        """加载单张图像，异常时返回全0张量"""
        try:
            with PILImage.open(img_path) as img:
                return self.transform(img.convert('RGB'))
        except:
            return torch.zeros(3, self.image_size, self.image_size)

    def __getitem__(self, idx):
        self.access_count += 1
        if idx < len(self.image_files):
            img_path = self.image_files[idx]
            # 标签和类别推断逻辑一致
            parent_dir = os_path.dirname(img_path)
            class_name = basename(parent_dir)
            if os_path.relpath(parent_dir, self.folder_path) == '.':
                class_name = '默认'
            label = self.classes.get(class_name, 0)
            # 优先读缓存
            cached = self.cache.get(img_path)
            if cached is not None:
                self.cache_hit_count += 1
                image = cached.clone()
            else:
                image = self._load_image(img_path)
                self.cache.put(img_path, image)
        else:
            # 超范围返回填充样本
            image = torch.zeros(3, self.image_size, self.image_size)
            label = 0
        return image, label

    def clear_cache(self):
        """清空LRU缓存"""
        self.cache.clear()

    def get_cache_stats(self):
        """获取缓存命中率等统计信息"""
        if self.access_count == 0:
            return {'hit_rate': 0, 'cache_size': 0}
        return {
            'hit_rate': self.cache_hit_count / self.access_count,
            'cache_size': len(self.cache.cache),
            'access_count': self.access_count
        }

class TextDataset(Dataset):
    """生成式因果语言模型数据集，按字分词，适配中文，保存显式词表"""
    def __init__(self, folder_path, vocab_size=1000, seq_len=128, 
                 progress_callback=None, scan_interval=100, pad_token_id=0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.scan_interval = scan_interval
        self.texts = []
        
        # 扫描文本文件
        print(f"[TextDataset] 开始扫描生成式训练数据: {folder_path}")
        if path_exists(folder_path):
            self._scan_with_progress(folder_path, progress_callback)
        
        if not self.texts:
            raise ValueError("未扫描到任何有效.txt文件，请检查数据路径")
        print(f"[TextDataset] 共 {len(self.texts)} 个训练样本")

        # -------------------------- 新增：构建并保存显式词表，训练测试共用 --------------------------
        # 收集所有出现的字符（按字分词）
        char_set = set()
        for text in self.texts:
            char_set.update(list(text))  # 按字分词；如果要按词分词，这里改成 text.split()
        # 排序后分配固定ID，保证同一个字每次都是同一个ID，不依赖hash随机性
        self.char2token = {char: idx for idx, char in enumerate(sorted(char_set))}
        self.token2char = {idx: char for char, idx in self.char2token.items()}
        # 如果词表长度超过设置的vocab_size，提示裁剪
        if len(self.char2token) > vocab_size:
            print(f"[TextDataset] 警告：实际词表大小{len(self.char2token)}超过设定{vocab_size}，将保留前{vocab_size}个常用字")
            self.char2token = dict(list(self.char2token.items())[:vocab_size])
            self.token2char = {idx: char for char, idx in self.char2token.items()}
        # 保存词表到数据集同目录，测试时加载
        vocab_save_path = f"{folder_path}_token2char.pth"
        torch.save(self.token2char, vocab_save_path)
        print(f"[TextDataset] 词表已保存到: {vocab_save_path}，词表大小: {len(self.token2char)}")
        # -----------------------------------------------------------------------

    def _scan_with_progress(self, folder_path, callback):
        """递归扫描所有txt文件"""
        count = 0
        start = time_now()
        for root, _, files in walk(folder_path):
            for f in files:
                if f.endswith('.txt'):
                    try:
                        with open(path_join(root, f), 'r', encoding='utf-8', errors='ignore') as fp:
                            t = fp.read().strip()
                            if t:
                                self.texts.append(t)
                                count += 1
                                if count % self.scan_interval == 0 and callback:
                                    callback(count, '?', f'扫描文本中... 已发现 {count} 个样本 ({time_now()-start:.0f}s)')
                    except:
                        pass
        if callback:
            callback(count, count, f'扫描完成，共 {count} 个样本 ({time_now()-start:.0f}s)')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # -------------------------- 修改：按字分词，用显式词表映射，不用hash --------------------------
        tokens = [self.char2token.get(char, self.char2token.get(list(self.char2token.keys())[0], 0)) for char in text]
        # -----------------------------------------------------------------------

        # 截断/填充到固定长度
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.pad_token_id] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]
        
        # 生成式核心：输入为完整序列，标签为输入右移一位（预测下一个token）
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(tokens[1:] + [self.pad_token_id], dtype=torch.long)
        # 将pad位置的标签设为-100，训练时自动忽略
        labels[input_ids == self.pad_token_id] = -100
        return input_ids, labels

# -------------------------- 基础组件 --------------------------
class ConvBlock(nn.Module):
    """基础卷积块：卷积+批归一化+ReLU激活，用于ResNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MoELayer(nn.Module):
    """Mixture of Experts 层，替换标准 FFN，支持 Top-K 路由和负载均衡辅助损失"""
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1, aux_loss_weight=0.02):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.d_model = d_model

        # 门控网络（Router）
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 专家网络（每个专家是一个两层 FFN）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        返回: (batch_size, seq_len, d_model), aux_loss
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)

        # 门控得分
        gate_logits = self.gate(x_flat)  # (N, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (N, num_experts)

        # Top-K 选择
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 对每个专家执行计算
        for expert_idx in range(self.num_experts):
            # 哪些 token 选择了这个专家
            mask = (topk_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            expert_output = self.experts[expert_idx](expert_input)
            # 该 token 对应这个专家的权重
            weight = topk_weights[mask][:, topk_indices[mask].eq(expert_idx).float().argmax(dim=-1)]
            output[mask] += expert_output * weight.unsqueeze(-1)

        # 负载均衡辅助损失（鼓励各专家使用率均衡）
        # 每个专家被选中的概率
        if self.training:
            expert_usage = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                expert_usage[i] = (topk_indices == i).any(dim=-1).float().mean()
            # 理想分布：均匀分布
            target_usage = torch.ones_like(expert_usage) / self.num_experts
            aux_loss = F.mse_loss(expert_usage, target_usage) * self.aux_loss_weight
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return output.view(batch_size, seq_len, d_model), aux_loss

# -------------------------- 模型类 --------------------------
class SimpleResNet(nn.Module):
    """轻量级ResNet，用于图像分类任务"""
    def __init__(self, image_size=224, num_classes=2, in_channels=3, base_channels=64, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        
        # 基础卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 残差块层
        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels*2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, 2, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels*4, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ConvBlock(in_channels, out_channels, stride=stride) for stride in [stride]] + \
                [ConvBlock(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.fc(x)

class TextTransformerModel(nn.Module):
    """生成式Transformer大模型（Decoder-only架构），支持MLA和MoE"""
    def __init__(self, vocab_size=1000, d_model=512, n_layers=6, n_heads=8, 
                 d_ff=2048, max_seq_len=128, dropout=0.1, pad_token_id=0,
                 use_moe=False, moe_experts=4, moe_top_k=2, aux_loss_weight=0.02,
                 moe_noise_epsilon=0.01, use_mla=False, mla_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_moe = use_moe
        self.use_mla = use_mla
        self.pad_token_id = pad_token_id
        self.aux_loss_weight = aux_loss_weight

        # 词嵌入+位置嵌入（支持pad token）
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # 标准Decoder-only架构
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # MoE层（替换每层decoder的FFN，可选）
        if use_moe:
            self.moe_layers = nn.ModuleList([
                MoELayer(d_model, d_ff, num_experts=moe_experts, top_k=moe_top_k,
                         dropout=dropout, aux_loss_weight=aux_loss_weight)
                for _ in range(n_layers)
            ])
        else:
            self.moe_layers = None

        # LM头，支持MLA
        if use_mla:
            self.mla_proj = nn.Linear(d_model, mla_dim)
            self.lm_head = nn.Linear(mla_dim, vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight  # 权重共享

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx] = 0
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, return_aux_loss=False):
        """
        生成式前向传播，支持MoE
        input_ids: shape (batch_size, seq_len)
        返回：(batch_size, seq_len, vocab_size) 预测logits
        """
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool)
        )
        tgt_key_padding_mask = (input_ids == self.pad_token_id)

        x = self.decoder(
            x, x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # MoE 前向传播
        total_aux_loss = 0.0
        if self.use_moe and self.moe_layers is not None:
            for moe_layer in self.moe_layers:
                x, aux_loss = moe_layer(x)
                total_aux_loss += aux_loss

        if self.use_mla:
            x = self.mla_proj(x)
        logits = self.lm_head(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_aux_loss:
            return logits, total_aux_loss
        return logits

    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=50, pad_token_id=0, eos_token_id=None):
        """生成式推理方法，和原有逻辑一致"""
        self.eval()
        batch_size = prompt_ids.size(0)
        
        if prompt_ids.max() >= self.vocab_size or prompt_ids.min() < 0:
            raise ValueError(
                f"Prompt token超出词表范围！词表大小: {self.vocab_size}, "
                f"Prompt最大token: {prompt_ids.max().item()}, 最小token: {prompt_ids.min().item()}"
            )
        temperature = max(temperature, 1e-8)
        top_k = min(top_k, self.vocab_size) if top_k > 0 else 0

        generated = prompt_ids.clone()

        for step in range(max_new_tokens):
            input_seq = generated[:, -self.max_seq_len:] if generated.size(1) > self.max_seq_len else generated
            with torch.no_grad():
                logits = self.forward(input_seq)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise RuntimeError("模型输出logits包含nan/inf，参数损坏，请重新训练")
                next_token_logits = logits[:, -1, :] / temperature
                
                if top_k > 0:
                    valid_top_k = min(top_k, next_token_logits.size(-1))
                    topk_logits, topk_indices = torch.topk(next_token_logits, valid_top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, topk_indices, topk_logits)

                probs = torch.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    probs = torch.ones_like(probs) / self.vocab_size
                else:
                    prob_sum = probs.sum(dim=-1, keepdim=True)
                    probs = torch.where(prob_sum > 0, probs / prob_sum, torch.ones_like(probs) / self.vocab_size)

                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated