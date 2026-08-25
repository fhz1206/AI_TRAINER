"""
trainers.diffusion — 扩散模型训练器

- train_diffusion      : 无条件图像生成（DDPM ε-MSE）
- train_diffusion_edit : 图像编辑适配（条件拼接 + 退化对自监督）

数据集直接用 ImageDataset 读图，取值域缩放到 [-1,1]；
训练目标为噪声预测 MSE（无分类准确率概念，进度以 loss 呈现）。
"""
import traceback
from os import makedirs
from os.path import join as path_join, getsize
from time import time as time_now

import torch
from torch.optim import AdamW

from model import ImageDataset
from model_zoo.diffusion import DiffusionModel, DiffusionEditModel
from model_io import save_model
from database import save_model_record

from .common import update_task, build_loader, maybe_compile, fail_task


def _iter_images(loader):
    """ImageDataset 返回 (img[0,1], label)，这里只要图像并压到 [-1,1]"""
    for imgs, _ in loader:
        yield imgs * 2.0 - 1.0


def _save_diffusion(model, key, arch, user_id, task_id, model_params, train_params,
                    final_loss, thread_start, training_tasks, prefix):
    """保存产物 + 数据库记录 + 完成状态（两条扩散链路共用）"""
    user_dir = f'models/{user_id}'
    makedirs(user_dir, exist_ok=True)
    model_name = f'{prefix}_{task_id}_{int(time_now())}.safetensors'
    model_path = path_join(user_dir, model_name)

    object.__setattr__(model, '_metadata', {
        'model_type': key,
        'architecture': arch,
        'model_params': model_params,
        'train_params': train_params,
        'final_loss': round(final_loss, 6),
        'total_time': round(time_now() - thread_start, 2),
        'description': f'{arch} | 最终损失: {final_loss:.4f}',
    })

    update_task(training_tasks, task_id, progress=98, message='⏳ 写入磁盘...')
    save_model(model, model_path)
    file_size = getsize(model_path)
    total_time = time_now() - thread_start

    save_model_record(user_id=user_id, model_name=model_name,
                      model_type='diffusion' if key == 'image_diffusion' else 'diffusion_edit',
                      file_size=file_size, file_path=model_path,
                      accuracy=None, loss=round(final_loss, 6),
                      epochs=train_params['epochs'])

    update_task(training_tasks, task_id,
                status='completed', progress=100,
                loss=round(final_loss, 6),
                message=(f'✅ 训练完成！最终损失: {final_loss:.4f} | '
                         f'耗时: {total_time:.0f}s | 模型文件: {model_name}'))
    print(f"========== 🎨 {arch} 训练完成 (总耗时: {total_time:.0f}s) ==========\n")


def train_diffusion(user_id, task_id, model_params, train_params, training_tasks):
    """无条件图像生成扩散模型训练"""
    thread_start = time_now()
    try:
        print(f"\n========== 🎨 扩散图像生成训练开始 task_id={task_id} ==========")
        training_tasks[task_id] = {
            'status': 'running', 'progress': 0, 'loss': None,
            'accuracy': None, 'message': '🔄 初始化训练线程...'
        }

        dataset = ImageDataset(
            train_params['data_path'],
            image_size=model_params.get('image_size', 32),
            cache_capacity=min(model_params.get('cache_capacity', 300), 1000),
            scan_interval=50,
        )
        # 扩散对类别不敏感；把 num_classes 归一避免误导前端
        model_params['num_classes'] = dataset.num_classes
        print(f"[Diffusion] 数据集: 共{len(dataset)}个样本")

        update_task(training_tasks, task_id, progress=20, message='🧠 构建扩散UNet...')
        model = DiffusionModel(
            image_size=model_params.get('image_size', 32),
            base_channels=model_params.get('base_channels', 32),
            num_timesteps=model_params.get('num_timesteps', 300),
            attn_heads=model_params.get('attn_heads', 4),
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[Diffusion] 模型参数量: {param_count:,}")
        run_model = maybe_compile(model, 'Diffusion')

        update_task(training_tasks, task_id, progress=25, message='📦 准备数据加载器...')
        loader = build_loader(dataset, train_params.get('batch_size', 16),
                              tag='Diffusion')
        if len(loader) == 0:
            raise ValueError("数据集样本数小于batch_size，请减小batch_size或增加数据")

        opt = AdamW(model.parameters(), lr=train_params.get('learning_rate', 2e-4))
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0
        total_loss = 0.0

        update_task(training_tasks, task_id, progress=26, message='🚀 训练启动...')
        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            model.train()
            for batch_idx, x0 in enumerate(_iter_images(loader)):
                opt.zero_grad()
                loss = run_model(x0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_params.get('max_grad_norm', 1.0))
                opt.step()
                total_loss += loss.item()
                batch_count += 1
                progress = min(26 + int((batch_count / max(total_batches, 1)) * 69), 95)
                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(loss.item(), 6),
                    'accuracy': None,
                    'message': (f'Epoch {epoch+1}/{train_params["epochs"]} '
                                f'| Batch {batch_idx+1}/{len(loader)} '
                                f'| Noise-MSE: {loss.item():.4f}')
                })
            print(f"[Diffusion] Epoch {epoch+1} | Loss: {total_loss/max(len(loader),1):.4f}"
                  f" | 耗时: {time_now()-epoch_start:.1f}s")

        final_loss = total_loss / max(len(loader) * 1, 1)
        _save_diffusion(model, 'image_diffusion', 'Diffusion-UNet(DDPM)',
                        user_id, task_id, model_params, train_params,
                        final_loss, thread_start, training_tasks, 'dif_gen')

    except Exception as e:
        fail_task(training_tasks, task_id, e, traceback.format_exc())


def train_diffusion_edit(user_id, task_id, model_params, train_params, training_tasks):
    """图像编辑适配扩散模型训练（退化对自监督）"""
    thread_start = time_now()
    try:
        print(f"\n========== 🖌️ 扩散图像编辑训练开始 task_id={task_id} ==========")
        training_tasks[task_id] = {
            'status': 'running', 'progress': 0, 'loss': None,
            'accuracy': None, 'message': '🔄 初始化训练线程...'
        }

        dataset = ImageDataset(
            train_params['data_path'],
            image_size=model_params.get('image_size', 32),
            cache_capacity=min(model_params.get('cache_capacity', 300), 1000),
            scan_interval=50,
        )
        model_params['num_classes'] = dataset.num_classes
        print(f"[EditDiffusion] 数据集: 共{len(dataset)}个样本")

        update_task(training_tasks, task_id, progress=20, message='🧠 构建编辑适配UNet...')
        model = DiffusionEditModel(
            image_size=model_params.get('image_size', 32),
            base_channels=model_params.get('base_channels', 32),
            num_timesteps=model_params.get('num_timesteps', 300),
            attn_heads=model_params.get('attn_heads', 4),
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[EditDiffusion] 模型参数量: {param_count:,}")
        run_model = maybe_compile(model, 'EditDiffusion')

        update_task(training_tasks, task_id, progress=25, message='📦 准备数据加载器...')
        loader = build_loader(dataset, train_params.get('batch_size', 16),
                              tag='EditDiffusion')
        if len(loader) == 0:
            raise ValueError("数据集样本数小于batch_size，请减小batch_size或增加数据")

        opt = AdamW(model.parameters(), lr=train_params.get('learning_rate', 2e-4))
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0
        total_loss = 0.0

        update_task(training_tasks, task_id, progress=26, message='🚀 训练启动...')
        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            model.train()
            for batch_idx, x0 in enumerate(_iter_images(loader)):
                cond = model.make_condition(x0)
                opt.zero_grad()
                loss = run_model(cond, x0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_params.get('max_grad_norm', 1.0))
                opt.step()
                total_loss += loss.item()
                batch_count += 1
                progress = min(26 + int((batch_count / max(total_batches, 1)) * 69), 95)
                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(loss.item(), 6),
                    'accuracy': None,
                    'message': (f'Epoch {epoch+1}/{train_params["epochs"]} '
                                f'| Batch {batch_idx+1}/{len(loader)} '
                                f'| Noise-MSE: {loss.item():.4f}')
                })
            print(f"[EditDiffusion] Epoch {epoch+1} | Loss: {total_loss/max(len(loader),1):.4f}"
                  f" | 耗时: {time_now()-epoch_start:.1f}s")

        final_loss = total_loss / max(len(loader), 1)
        _save_diffusion(model, 'image_edit_diffusion', 'Diffusion-UNet(Edit-Adapter)',
                        user_id, task_id, model_params, train_params,
                        final_loss, thread_start, training_tasks, 'dif_edit')

    except Exception as e:
        fail_task(training_tasks, task_id, e, traceback.format_exc())
