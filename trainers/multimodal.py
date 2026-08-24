"""
trainers.multimodal — 多模态单流模型训练器

图文配对训练：图片目录下每张图配同名 .txt 描述文本（无配对的图用占位描述），
图像 patch 与文本 token 拼成单流序列，做「看图续写」式下一 token 预测。
"""
import os
import random
import traceback
from os import makedirs
from os.path import join as path_join, getsize, basename, splitext
from time import time as time_now

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import AdamW

from models.multimodal import MultiModalSingleStream
from model_io import save_model, save_vocab_json
from database import save_model_record

from .common import update_task, build_loader, maybe_compile, fail_task


class ImageTextPairDataset(Dataset):
    """
    图文配对数据集：
    - 扫描目录下所有图片；同目录找同名 .txt 作为描述，缺失则用「一张照片」占位
    - 字符级词表（与 TextDataset 同风格），JSON 落盘供测试端复用
    - 返回 (image[-1,1], text_ids)
    """

    def __init__(self, folder_path, vocab_size=1000, max_txt_len=24,
                 image_size=32, pad_token_id=0):
        self.folder_path = folder_path
        self.image_size = image_size
        self.max_txt_len = max_txt_len
        self.pad_token_id = pad_token_id

        # ---- 收集图文对 ----
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        pairs = []
        for root, dirs, files in os.walk(folder_path):
            for f in sorted(files):
                if f.lower().endswith(exts):
                    img_p = path_join(root, f)
                    txt_p = splitext(img_p)[0] + '.txt'
                    try:
                        raw = np.fromfile(txt_p, dtype=np.uint8)
                        desc = raw.tobytes().decode('utf-8', 'ignore').strip()
                    except OSError:
                        desc = ''
                    pairs.append((img_p, desc or '一张照片'))
        if not pairs:
            raise ValueError(f"目录中没有找到图片文件：{folder_path}")
        self.pairs = pairs
        print(f"[MultiModal] 图文对数量: {len(self.pairs)}")

        # ---- 字符级词表 ----
        from collections import Counter
        freq = Counter(ch for _, d in pairs for ch in d)
        ranked = [c for c, _ in freq.most_common(vocab_size - 3)]
        special = ['<pad>', '<unk>', '<img>']
        vocab = special + ranked[:vocab_size - len(special)]
        self.char2token = {c: i for i, c in enumerate(vocab)}
        self.token2char = {i: c for c, i in self.char2token.items()}
        save_vocab_json(self.token2char,
                        path_join(folder_path, '_multimodal_pairs'))
        print(f"[MultiModal] 词表大小: {len(self.char2token)}")

    def encode(self, text):
        ids = [self.char2token.get(c, 1) for c in text[:self.max_txt_len]]
        if len(ids) < self.max_txt_len:
            ids += [self.pad_token_id] * (self.max_txt_len - len(ids))
        return ids[:self.max_txt_len]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, desc = self.pairs[idx]
        buf = np.fromfile(img_p, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((self.image_size, self.image_size, 3), 128, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = torch.from_numpy(img.transpose(2, 0, 1)) * 2.0 - 1.0   # [-1,1]
        return t, torch.tensor(self.encode(desc), dtype=torch.long)


def train_multimodal(user_id, task_id, model_params, train_params, training_tasks):
    """多模态单流模型全流程"""
    thread_start = time_now()
    try:
        print(f"\n========== 🧩 多模态单流训练开始 task_id={task_id} ==========")
        print(f"模型参数: {model_params}")
        print(f"训练参数: {train_params}")

        training_tasks[task_id] = {
            'status': 'running', 'progress': 0, 'loss': None,
            'accuracy': None, 'message': '🔄 初始化训练线程...'
        }

        update_task(training_tasks, task_id, message='🔍 扫描图文配对数据...')
        dataset = ImageTextPairDataset(
            train_params['data_path'],
            vocab_size=model_params.get('vocab_size', 1000),
            max_txt_len=model_params.get('max_txt_len', 24),
            image_size=model_params.get('image_size', 32),
            pad_token_id=model_params.get('pad_token_id', 0),
        )
        model_params['num_classes'] = dataset.__len__()

        update_task(training_tasks, task_id, progress=20, message='🧠 构建多模态单流模型...')
        model = MultiModalSingleStream(
            vocab_size=model_params.get('vocab_size', 1000),
            image_size=model_params.get('image_size', 32),
            patch_size=model_params.get('patch_size', 8),
            d_model=model_params.get('d_model', 192),
            n_layers=model_params.get('n_layers', 4),
            n_heads=model_params.get('n_heads', 4),
            d_ff=model_params.get('d_ff', 384),
            max_seq_len=model_params.get('max_seq_len', 64),
            dropout=model_params.get('dropout', 0.1),
            pad_token_id=model_params.get('pad_token_id', 0),
            use_moe=model_params.get('use_moe', False),
            moe_experts=model_params.get('moe_experts', 4),
            moe_top_k=model_params.get('moe_top_k', 2),
            attention_type=model_params.get('attention_type', 'flash'),
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[MultiModal] 模型参数量: {param_count:,}")
        run_model = maybe_compile(model, 'MultiModal')

        update_task(training_tasks, task_id, progress=25, message='📦 准备数据加载器...')
        loader = build_loader(dataset, train_params.get('batch_size', 16),
                              tag='MultiModal')

        opt = AdamW(model.parameters(),
                    lr=min(train_params.get('learning_rate', 3e-4), 1e-3),
                    weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0

        update_task(training_tasks, task_id, progress=26, message='🚀 训练启动...')
        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            n_batches = 0
            correct = total = 0
            model.train()
            for batch_idx, (imgs, txt_ids) in enumerate(loader):
                # 输入去掉最后一位，目标左移一位（下一 token 预测）
                logits = run_model(imgs, txt_ids[:, :-1])
                gold = txt_ids[:, 1:]
                keep = (gold != dataset.pad_token_id)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 gold.reshape(-1))

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_loss += loss.item()
                n_batches += 1
                batch_count += 1
                with torch.no_grad():
                    pred = logits.argmax(-1)[keep]
                    hit = (pred == gold[keep]).sum().item()
                    denom = max(keep.sum().item(), 1)
                    correct += hit
                    total += denom

                acc = 100 * correct / max(total, 1)
                progress = min(26 + int((batch_count / max(total_batches, 1)) * 69), 95)
                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(loss.item(), 6),
                    'accuracy': round(acc, 4),
                    'message': (f'Epoch {epoch+1}/{train_params["epochs"]} '
                                f'| Batch {batch_idx+1}/{len(loader)} '
                                f'| Loss: {loss.item():.4f} | TokenAcc: {acc:.2f}%')
                })
            print(f"[MultiModal] Epoch {epoch+1} | Loss: {total_loss/max(n_batches,1):.4f}"
                  f" | 耗时: {time_now()-epoch_start:.1f}s")

        final_loss = total_loss / max(n_batches, 1)
        final_acc = 100 * correct / max(total, 1)

        # ---- 保存 ----
        user_dir = f'models/{user_id}'
        makedirs(user_dir, exist_ok=True)
        model_name = f'mm_stream_{task_id}_{int(time_now())}.safetensors'
        model_path = path_join(user_dir, model_name)

        object.__setattr__(model, '_metadata', {
            'model_type': 'multimodal_stream',
            'architecture': f'SingleStream-Decoder({model.attention_type} attention)',
            'model_params': model_params,
            'train_params': train_params,
            'final_loss': round(final_loss, 6),
            'final_accuracy': round(final_acc, 4),
            'total_time': round(time_now() - thread_start, 2),
            'description': (f'多模态单流模型 | 注意力: {model.attention_type} | '
                            f'TokenAcc: {final_acc:.2f}%'),
        })

        update_task(training_tasks, task_id, progress=98, message='⏳ 写入磁盘...')
        save_model(model, model_path)
        file_size = getsize(model_path)
        total_time = time_now() - thread_start

        save_model_record(user_id=user_id, model_name=model_name,
                          model_type='multimodal',
                          file_size=file_size, file_path=model_path,
                          accuracy=round(final_acc, 4),
                          loss=round(final_loss, 6),
                          epochs=train_params['epochs'])

        update_task(training_tasks, task_id,
                    status='completed', progress=100,
                    loss=round(final_loss, 6), accuracy=round(final_acc, 4),
                    message=(f'✅ 训练完成！Token准确率: {final_acc:.2f}% | '
                             f'损失: {final_loss:.4f} | 耗时: {total_time:.0f}s | '
                             f'模型文件: {model_name}'))
        print(f"========== 🧩 多模态训练完成 (总耗时: {total_time:.0f}s) ==========\n")

    except Exception as e:
        fail_task(training_tasks, task_id, e, traceback.format_exc())
