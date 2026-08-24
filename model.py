"""
model.py — 数据集工具 + 兼容门面

职责：
1. 数据管线：LRUCache / ImageDataset / TextDataset（全平台共用）
2. 兼容门面：重新导出 models 包中的模型类，保持既有导入路径可用
   （from model import SimpleResNet, TextTransformerModel），
   并让历史 .pth 检查点（pickle 按 model.* 路径找类）继续可反序列化。

新代码请直接从 models / architectures 包导入；本文件不再定义网络结构。
"""
import json
import os
from collections import OrderedDict
from os import walk, path as os_path
from os.path import basename
from threading import Lock
from time import time as time_now

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 兼容导出：结构不变的模型直接来自 models 包；TextTransformerModel 绑定旧版完整实现，
# 使历史 .pth 整对象检查点反序列化后仍走当年的 forward/generate（新版积木式实现在 models.text）
from models.vision import ConvBlock, SimpleResNet, ViTModel          # noqa: F401
from models.legacy import TextTransformerModel, MoELayer             # noqa: F401

import warnings
from random import randint, seed as random_seed

warnings.filterwarnings('ignore')
random_seed(42)


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

    def __getstate__(self):
        # Lock 不可跨进程序列化：交给 pickle 时剥离，载入侧重建
        state = self.__dict__.copy()
        state['lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = Lock()


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

        print(f"[ImageDataset] 开始扫描: {folder_path}")
        if os_path.exists(folder_path):
            self._scan_with_progress(folder_path, progress_callback)
        print(f"[ImageDataset] 共 {len(self.image_files)} 个文件")

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

        # ImageNet标准图像预处理（ToTensor 在前：兼容 OpenCV ndarray 输入）
        self.transform = Compose([
            ToTensor(),
            Resize((image_size, image_size)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.access_count = 0
        self.cache_hit_count = 0

    def _scan_with_progress(self, folder, callback):
        count = 0
        start = time_now()
        for root, dirs, files in walk(folder):
            dirs.sort()
            for f in sorted(files):
                ext = os_path.splitext(f)[1].lower()
                if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
                    self.image_files.append(os_path.join(root, f))
                    count += 1
                    if callback and count % self.scan_interval == 0:
                        total = len(self.image_files)
                        callback(count, '?',
                                 f'扫描中... 已发现 {count} 个文件 ({time_now()-start:.0f}s)')
        if callback:
            callback(count, count, f'扫描完成，共 {count} 个文件 ({time_now()-start:.0f}s)')

    def __len__(self):
        return max(len(self.image_files), 1)

    def clear_cache(self):
        """每个 epoch 调用：释放图像缓存，避免长训练占用过多内存"""
        self.cache.clear()

    def _load_image(self, img_path):
        """加载单张图像（OpenCV 解码，兼容中文路径），异常时返回全0张量"""
        try:
            buf = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                return torch.zeros(3, self.image_size, self.image_size)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(rgb)
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size)

    def __getitem__(self, idx):
        self.access_count += 1
        if idx < len(self.image_files):
            img_path = self.image_files[idx]
            parent_dir = os_path.dirname(img_path)
            class_name = basename(parent_dir)
            if os_path.relpath(parent_dir, self.folder_path) == '.':
                class_name = '默认'
            label = self.classes.get(class_name, 0)
            cached = self.cache.get(img_path)
            if cached is not None:
                self.cache_hit_count += 1
                image = cached.clone()
            else:
                image = self._load_image(img_path)
                self.cache.put(img_path, image)
        else:
            image = torch.zeros(3, self.image_size, self.image_size)
            label = 0
        return image, label


class TextDataset(Dataset):
    """
    文本生成数据集：滑窗切序列；显式词表 char->token 映射，
    词表以 JSON 存储（去 pickle），测试端经 model_io.load_vocab 加载。
    """

    def __init__(self, folder_path, vocab_size=1000, seq_len=128,
                 progress_callback=None, scan_interval=50, pad_token_id=0):
        self.folder_path = folder_path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        texts = []
        print(f"[TextDataset] 开始扫描: {folder_path}")
        if os_path.exists(folder_path):
            self._scan_with_progress(folder_path, texts, progress_callback)
        print(f"[TextDataset] 共读取 {len(texts)} 个文本文件")
        full_text = '\n'.join(texts)
        print(f"[TextDataset] 语料总字符数: {len(full_text)}")

        chars = sorted(set(full_text))
        print(f"[TextDataset] 语料字符种类: {len(chars)}")
        if len(chars) < 2:
            raise ValueError(
                f"语料字符过少({len(chars)}种)，无法构建词表。请检查训练数据目录是否包含有效txt文本"
            )

        # 高频字优先截断到 vocab_size
        from collections import Counter
        freq = Counter(full_text)
        ranked = [c for c, _ in freq.most_common(vocab_size - 2)]
        special = ['<pad>', '<unk>']
        vocab_chars = special + ranked[:vocab_size - len(special)]
        self.char2token = {ch: i for i, ch in enumerate(vocab_chars)}
        self.token2char = {i: ch for ch, i in self.char2token.items()}
        if len(self.char2token) > vocab_size:
            self.char2token = dict(list(self.char2token.items())[:vocab_size])
            self.token2char = {idx: ch for ch, idx in self.char2token.items()}

        # 词表保存为 JSON（避免 pickle）
        from model_io import save_vocab_json
        vocab_save_path = save_vocab_json(self.token2char, folder_path)
        print(f"[TextDataset] 词表已保存到: {vocab_save_path}，词表大小: {len(self.token2char)}")

        ids = [self.char2token.get(ch, 1) for ch in full_text]
        n_seq = max((len(ids) - seq_len) // seq_len + 1, 0) if len(ids) > seq_len else 0
        self.samples = []
        for i in range(n_seq):
            chunk = ids[i * seq_len:(i + 1) * seq_len]
            if len(chunk) == seq_len:
                self.samples.append(chunk)
        print(f"[TextDataset] 切分出 {len(self.samples)} 条长度为 {seq_len} 的样本")
        if not self.samples:
            raise ValueError(
                f"语料太短（{len(ids)} 字符），无法切出哪怕一条长度为 {seq_len} 的样本。"
                f"请增加文本量或调小 max_seq_len"
            )

    def _scan_with_progress(self, folder, out_texts, callback):
        count = 0
        start = time_now()

        def _read_one(p):
            try:
                raw = np.fromfile(p, dtype=np.uint8)
                text = raw.tobytes().decode('utf-8', errors='ignore')
                if text.strip():
                    out_texts.append(text)
                    return 1
            except Exception:
                pass
            return 0

        # 兼容直接传入单个 txt 文件的情况（上传单文件时 data_path 指向文件本身）
        if os_path.isfile(folder):
            count += _read_one(folder)
            if callback:
                callback(count, count, f'扫描完成，共 {count} 个文件 ({time_now()-start:.0f}s)')
            return

        for root, dirs, files in walk(folder):
            dirs.sort()
            for f in sorted(files):
                if f.lower().endswith('.txt'):
                    p = os_path.join(root, f)
                    count += _read_one(p)
                    if callback and count % self.scan_interval == 0:
                        callback(count, '?',
                                 f'扫描中... 已发现 {count} 个文件 ({time_now()-start:.0f}s)')
        if callback:
            callback(count, count, f'扫描完成，共 {count} 个文件 ({time_now()-start:.0f}s)')

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if idx < len(self.samples):
            chunk = self.samples[idx]
        else:
            chunk = [self.pad_token_id] * self.seq_len
        x = torch.tensor(chunk, dtype=torch.long)
        y = torch.roll(x, shifts=-1, dims=0)
        return x, y
