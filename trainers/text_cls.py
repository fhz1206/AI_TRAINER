"""
trainers.text_cls — 语言分类模型训练器

数据：顶层子文件夹名 = 类别，每个 .txt 为一条样本（TextClassificationDataset）。
模型：字符级 Transformer 双向编码器（支持积木式混合注意力）。
"""
import traceback
from os import makedirs
from os.path import join as path_join, getsize
from time import time as time_now

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import TextClassificationDataset
from model_zoo.text_cls import TextClassifier
from model_io import save_model
from database import save_model_record

from .common import update_task, build_loader, maybe_compile, fail_task


def train_text_classification(user_id, task_id, model_params,
                              train_params, training_tasks):
    """语言分类全流程：扫描数据集 → 构建模型 → 训练 → 保存 → 记录数据库"""
    thread_start = time_now()
    try:
        print(f"\n========== 📝 语言分类训练开始 task_id={task_id} ==========")
        print(f"模型参数: {model_params}")
        print(f"训练参数: {train_params}")

        training_tasks[task_id] = {
            'status': 'running', 'progress': 0, 'loss': None,
            'accuracy': None, 'message': '🔄 初始化训练线程...'
        }

        # ---- 数据集 ----
        def progress_cb(cur, total, msg):
            pct = 5 if isinstance(total, str) else min(int(cur / max(total, 1) * 20), 19)
            update_task(training_tasks, task_id, progress=pct, message=f'🔍 {msg}')

        dataset = TextClassificationDataset(
            train_params['data_path'],
            vocab_size=model_params.get('vocab_size', 5000),
            max_seq_len=model_params.get('max_seq_len', 64),
            pad_token_id=model_params.get('pad_token_id', 0),
            progress_callback=progress_cb,
        )
        model_params['num_classes'] = len(dataset.classes)
        print(f"[TextCls] 数据集统计: {len(dataset)} 样本 / "
              f"{len(dataset.classes)} 类")

        # ---- 模型 ----
        update_task(training_tasks, task_id, progress=20,
                    message='🧠 构建语言分类模型...')
        model = TextClassifier(
            vocab_size=dataset.vocab_size,
            num_classes=len(dataset.classes),
            d_model=model_params.get('d_model', 128),
            n_layers=model_params.get('n_layers', 4),
            n_heads=model_params.get('n_heads', 4),
            d_ff=model_params.get('d_ff', 256),
            max_seq_len=model_params.get('max_seq_len', 64),
            dropout=model_params.get('dropout', 0.1),
            pad_token_id=model_params.get('pad_token_id', 0),
            attention_type=model_params.get('attention_type', 'flash'),
            attention_plan=model_params.get('attention_plan'),
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[TextCls] 模型参数量: {param_count:,} | "
              f"注意力: {model.attention_summary}")
        run_model = maybe_compile(model, 'TextCls')

        # ---- 数据加载器 ----
        update_task(training_tasks, task_id, progress=25,
                    message='📦 准备数据加载器...')
        loader = build_loader(dataset, train_params.get('batch_size', 16),
                              tag='TextCls')
        if len(loader) == 0:
            raise ValueError("数据集样本数小于batch_size，请减小batch_size或增加数据")

        # ---- 训练循环 ----
        opt = AdamW(model.parameters(),
                    lr=train_params.get('learning_rate', 1e-3), weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0
        correct = total = 0

        update_task(training_tasks, task_id, progress=26, message='🚀 训练启动...')

        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            n_batches = 0
            correct = total = 0
            model.train()

            for batch_idx, (ids, labels) in enumerate(loader):
                opt.zero_grad()
                outputs = run_model(ids)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_params.get('max_grad_norm', 1.0))
                opt.step()

                total_loss += loss.item()
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                batch_count += 1

                acc = 100. * correct / max(total, 1)
                progress = min(26 + int((batch_count / max(total_batches, 1)) * 69), 95)
                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(loss.item(), 6),
                    'accuracy': round(acc, 4),
                    'message': (f'Epoch {epoch+1}/{train_params["epochs"]} '
                                f'| Batch {batch_idx+1}/{len(loader)} '
                                f'| Loss: {loss.item():.4f} | Acc: {acc:.2f}%')
                })
            avg_loss = total_loss / max(len(loader), 1)
            print(f"[TextCls] Epoch {epoch+1}/{train_params['epochs']} | "
                  f'Loss: {avg_loss:.4f} | 耗时: {time_now()-epoch_start:.1f}s')

        final_acc = 100. * correct / max(total, 1)
        final_loss = total_loss / max(len(loader), 1)

        # ---- 保存 ----
        update_task(training_tasks, task_id, progress=96, message='💾 正在保存模型...')
        user_dir = f'models/{user_id}'
        makedirs(user_dir, exist_ok=True)
        model_name = f'text_cls_{task_id}_{int(time_now())}.safetensors'
        model_path = path_join(user_dir, model_name)

        object.__setattr__(model, '_metadata', {
            'model_type': 'text_classifier',
            'architecture': f'TransformerEncoder({model.attention_summary})',
            'model_params': model_params,
            'train_params': train_params,
            'final_accuracy': round(final_acc, 4),
            'final_loss': round(final_loss, 6),
            'total_time': round(time_now() - thread_start, 2),
            'description': (f'语言分类模型 | 准确率: {final_acc:.2f}% | '
                            f'类别: {list(dataset.classes)}'),
        })

        update_task(training_tasks, task_id, progress=98, message='⏳ 写入磁盘...')
        save_model(model, model_path)
        file_size = getsize(model_path)
        total_time = time_now() - thread_start

        save_model_record(user_id=user_id, model_name=model_name,
                          model_type='text_cls',
                          file_size=file_size, file_path=model_path,
                          accuracy=round(final_acc, 4),
                          loss=round(final_loss, 6),
                          epochs=train_params['epochs'])

        update_task(training_tasks, task_id,
                    status='completed', progress=100,
                    loss=round(final_loss, 6), accuracy=round(final_acc, 4),
                    message=(f'✅ 训练完成！准确率: {final_acc:.2f}% | '
                             f'损失: {final_loss:.4f} | 耗时: {total_time:.0f}s | '
                             f'模型文件: {model_name}'))
        print(f"========== 📝 语言分类训练完成 (总耗时: {total_time:.0f}s) ==========\n")

    except Exception as e:
        fail_task(training_tasks, task_id, e, traceback.format_exc())
