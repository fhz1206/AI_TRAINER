"""
trainers.image_cls — 图像分类训练器（CNN 与 ViT 统一入口）

通过 model_key 区分架构：
- image_cnn : SimpleResNet 卷积网络
- image_vit : ViT（支持积木式注意力 full/flash/linear）
两者共用同一套数据管线与训练循环，仅模型构建参数不同。
"""
import traceback
from os import makedirs
from os.path import join as path_join, getsize
from time import time as time_now

import torch
import torch.nn as nn
from torch.optim import AdamW

from model import ImageDataset
from model_zoo import build_model
from model_io import save_model
from database import save_model_record

from .common import update_task, build_loader, maybe_compile, fail_task

_ARCH_INFO = {
    'image_cnn': {'architecture': 'SimpleResNet', 'label': 'CNN', 'prefix': 'cnn',
                  'db_type': 'cnn'},
    'image_vit': {'architecture': 'VisionTransformer', 'label': 'ViT', 'prefix': 'vit',
                  'db_type': 'vit'},
}


def _build_arch_kwargs(key, mp):
    if key == 'image_vit':
        return dict(
            image_size=mp.get('image_size', 224),
            patch_size=mp.get('patch_size', 32),
            in_channels=mp.get('in_channels', 3),
            num_classes=mp['num_classes'],
            d_model=mp.get('d_model', 192),
            n_layers=mp.get('n_layers', 6),
            n_heads=mp.get('n_heads', 4),
            d_ff=mp.get('d_ff', 384),
            dropout=mp.get('dropout', 0.1),
            attention_type=mp.get('attention_type', 'flash'),
        )
    return dict(
        image_size=mp.get('image_size', 224),
        num_classes=mp['num_classes'],
        in_channels=mp.get('in_channels', 3),
        base_channels=mp.get('base_channels', 64),
        dropout=mp.get('dropout', 0.1),
    )


def train_image_classification(model_key, user_id, task_id,
                               model_params, train_params, training_tasks):
    """图像分类全流程：扫描数据集 → 构建模型 → 训练 → 保存 → 记录数据库"""
    key = (model_key or 'image_cnn').lower()
    info = _ARCH_INFO[key]
    label = info['label']
    thread_start = time_now()

    try:
        print(f"\n========== 🖼️ {label} 图像分类训练开始 task_id={task_id} ==========")
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

        update_task(training_tasks, task_id, message='🔍 开始扫描图片数据集...')
        dataset = ImageDataset(
            train_params['data_path'],
            image_size=model_params.get('image_size', 224),
            cache_capacity=min(model_params.get('cache_capacity', 500), 2000),
            progress_callback=progress_cb,
            scan_interval=50,
        )
        model_params['num_classes'] = dataset.num_classes
        print(f"[{label}] 数据集统计: {dataset.num_classes}类, 共{len(dataset)}个样本")

        # ---- 模型 ----
        update_task(training_tasks, task_id, progress=20, message=f'🧠 构建{label}模型...')
        model = build_model(key, **_build_arch_kwargs(key, model_params))
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[{label}] 模型参数量: {param_count:,}")
        run_model = maybe_compile(model, label)

        # ---- 数据加载器 ----
        update_task(training_tasks, task_id, progress=25, message='📦 准备数据加载器...')
        loader = build_loader(dataset, train_params['batch_size'], tag=label)
        if len(loader) == 0:
            raise ValueError(
                f"数据集样本数({len(dataset)})小于batch_size({train_params['batch_size']})，"
                f"请减小batch_size或增加训练数据")
        print(f"[{label}] DataLoader: 每轮{len(loader)}个batch")

        # ---- 训练循环 ----
        opt = AdamW(model.parameters(), lr=train_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0
        correct = total = 0
        total_loss = 0.0

        update_task(training_tasks, task_id, progress=26, message='🚀 训练启动...')

        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            correct = total = 0
            dataset.clear_cache()

            for batch_idx, (imgs, labels) in enumerate(loader):
                opt.zero_grad()
                outputs = run_model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                batch_count += 1

                progress = min(26 + int((batch_count / max(total_batches, 1)) * 69), 95)
                accuracy = 100. * correct / max(total, 1)
                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(loss.item(), 6),
                    'accuracy': round(accuracy, 4),
                    'message': (
                        f'Epoch {epoch+1}/{train_params["epochs"]} '
                        f'| Batch {batch_idx+1}/{len(loader)} '
                        f'| Loss: {loss.item():.4f} | Acc: {accuracy:.2f}%')
                })

            avg_loss = total_loss / len(loader)
            avg_acc = 100. * correct / max(total, 1)
            print(f"[{label}] Epoch {epoch+1}/{train_params['epochs']} | "
                  f'Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | '
                  f'耗时: {time_now()-epoch_start:.1f}s')

        final_acc = 100. * correct / max(total, 1)
        final_loss = total_loss / len(loader)

        # ---- 保存 ----
        update_task(training_tasks, task_id, progress=96, message='💾 正在保存模型...')
        user_dir = f'models/{user_id}'
        makedirs(user_dir, exist_ok=True)
        model_name = f"{info['prefix']}_{task_id}_{int(time_now())}.safetensors"
        model_path = path_join(user_dir, model_name)

        object.__setattr__(model, '_metadata', {
            'model_type': key,
            'architecture': info['architecture'],
            'model_params': model_params,
            'train_params': train_params,
            'final_accuracy': round(final_acc, 4),
            'final_loss': round(final_loss, 6),
            'total_time': round(time_now() - thread_start, 2),
            'description': f'{label}图像分类模型 | 准确率: {final_acc:.2f}%'
        })

        update_task(training_tasks, task_id, progress=98, message='⏳ 写入磁盘...')
        save_model(model, model_path)
        file_size = getsize(model_path)
        total_time = time_now() - thread_start
        print(f"[{label}] ✅ 模型已保存: {model_path} ({file_size/1024/1024:.1f} MB)")

        save_model_record(
            user_id=user_id, model_name=model_name, model_type=info['db_type'],
            file_size=file_size, file_path=model_path,
            accuracy=round(final_acc, 4), loss=round(final_loss, 6),
            epochs=train_params['epochs'])

        update_task(training_tasks, task_id,
                    status='completed', progress=100,
                    loss=round(final_loss, 6), accuracy=round(final_acc, 4),
                    message=(f'✅ 训练完成！准确率: {final_acc:.2f}% | '
                             f'损失: {final_loss:.4f} | 耗时: {total_time:.0f}s | '
                             f'模型文件: {model_name}'))
        print(f"========== 🖼️ {label} 训练完成 (总耗时: {total_time:.0f}s) ==========\n")

    except Exception as e:
        fail_task(training_tasks, task_id, e, traceback.format_exc())
