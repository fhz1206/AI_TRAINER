"""
trainer.py — 统一训练器
支持 CNN 图片训练 + 生成式文本大模型训练（兼容MoE/MLA）
训练完成后自动写入 SQLite3 数据库
适配拆分后的项目结构，线程安全，支持进度实时反馈
"""
import os
import random
import time
import traceback
import warnings
import math
from os import makedirs
from os.path import join as path_join, getsize
from random import seed as random_seed
from time import time as time_now

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 全局配置
warnings.filterwarnings('ignore')
random_seed(42)  # 固定随机种子保证训练可复现

# ==================== 项目模块导入 ====================
from database import save_model_record
from state import training_tasks
from model import SimpleResNet, TextTransformerModel, ImageDataset, TextDataset
from model_io import save_model


def _dataloader_workers():
    """DataLoader 工作进程数：半数CPU、上限4；可用 AITPP_DATALOADER_WORKERS 覆盖（0=单进程）"""
    env = os.environ.get('AITPP_DATALOADER_WORKERS')
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def _multiproc_loader_ok():
    """探测当前进程能否 spawn 子进程（打包 exe / stdin 等受限场景返回 False）"""
    import multiprocessing as mp
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        p = mp.Process(target=_noop_probe)
        p.start()
        p.join(timeout=15)
        return p.exitcode == 0
    except Exception:
        return False


def _noop_probe():
    pass

# ==================== 工具函数：线程安全更新任务状态 ====================
def update_task(tasks, task_id, **kwargs):
    """线程安全地更新全局训练任务状态"""
    if task_id in tasks:
        tasks[task_id].update(kwargs)

# ==================== 图片训练（CNN + SimpleResNet，完全保留原有逻辑） ====================
def train_image_model(user_id, task_id, model_params, train_params, training_tasks):
    """图像分类训练全流程：数据集加载→模型构建→训练→保存→数据库记录"""
    thread_start = time_now()

    try:
        print(f"\n========== 🖼️ CNN 训练开始 task_id={task_id} ==========")
        print(f"模型参数: {model_params}")
        print(f"训练参数: {train_params}")

        # ===== 第0步：写入初始状态 =====
        training_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'loss': None,
            'accuracy': None,
            'message': '🔄 初始化训练线程...'
        }

        # ===== 第1步：加载数据集（带进度回调）=====
        def progress_cb(cur, total, msg):
            if isinstance(total, str):
                scan_pct = 5
            else:
                scan_pct = min(int(cur / max(total, 1) * 20), 19)
            
            update_task(training_tasks, task_id,
                       progress=scan_pct,
                       message=f'🔍 {msg}')

        update_task(training_tasks, task_id, message='🔍 开始扫描图片数据集...')

        dataset = ImageDataset(
            train_params['data_path'],
            image_size=model_params.get('image_size', 224),
            cache_capacity=min(model_params.get('cache_capacity', 500), 2000),
            progress_callback=progress_cb,
            scan_interval=50
        )
        model_params['num_classes'] = dataset.num_classes
        print(f"[CNN] 数据集统计: {dataset.num_classes}类, 共{len(dataset)}个样本")

        # ===== 第2步：构建SimpleResNet模型 =====
        update_task(training_tasks, task_id,
                   progress=20,
                   message='🧠 构建CNN模型...')

        model = SimpleResNet(
            image_size=model_params.get('image_size', 224),
            num_classes=model_params['num_classes'],
            in_channels=model_params.get('in_channels', 3),
            base_channels=model_params.get('base_channels', 64),
            dropout=model_params.get('dropout', 0.1)
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[CNN] 模型参数量: {param_count:,}")

        # 可选加速：torch.compile（设环境变量 AITPP_TORCH_COMPILE=1 开启；失败自动回退）
        # 前向走 run_model；保存/评估始终用原始 model，避免编译包装层改变权重键名
        run_model = model
        if os.environ.get('AITPP_TORCH_COMPILE') == '1':
            try:
                run_model = torch.compile(model)
                print("[CNN] torch.compile 已启用")
            except Exception as e:
                print(f"[CNN] torch.compile 启用失败，使用 eager 模式: {e}")

        # ===== 第3步：构建DataLoader =====
        update_task(training_tasks, task_id,
                   progress=25,
                   message='📦 准备数据加载器...')

        _workers = _dataloader_workers()
        if _workers > 0 and not _multiproc_loader_ok():
            _workers = 0
            print("[CNN] 多进程加载不可用，回退单进程")
        loader = DataLoader(
            dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=_workers,
            pin_memory=True,
            persistent_workers=_workers > 0,
            drop_last=True
        )

        if len(loader) == 0:
            raise ValueError(
                f"数据集样本数({len(dataset)})小于batch_size({train_params['batch_size']})，"
                f"请减小batch_size或增加训练数据"
            )
        print(f"[CNN] DataLoader: 每轮{len(loader)}个batch")

        # ===== 第4步：训练循环 =====
        opt = AdamW(model.parameters(), lr=train_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0

        update_task(training_tasks, task_id,
                   progress=26,
                   message='🚀 训练启动...')

        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            correct = 0
            total = 0
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

                progress = 26 + int((batch_count / max(total_batches, 1)) * 69)
                progress = min(progress, 95)
                accuracy = 100. * correct / max(total, 1)

                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(loss.item(), 6),
                    'accuracy': round(accuracy, 4),
                    'message': (
                        f'Epoch {epoch+1}/{train_params["epochs"]} '
                        f'| Batch {batch_idx+1}/{len(loader)} '
                        f'| Loss: {loss.item():.4f} | Acc: {accuracy:.2f}%'
                    )
                })

            epoch_time = time_now() - epoch_start
            avg_loss = total_loss / len(loader)
            avg_acc = 100. * correct / max(total, 1)
            print(f"[CNN] Epoch {epoch+1}/{train_params['epochs']} | "
                  f'Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}% | '
                  f'耗时: {epoch_time:.1f}s')

        final_acc = 100. * correct / max(total, 1)
        final_loss = total_loss / len(loader)

        # ===== 第5步：保存模型 =====
        update_task(training_tasks, task_id,
                   progress=96,
                   message='💾 正在保存模型...')

        user_dir = f'models/{user_id}'
        makedirs(user_dir, exist_ok=True)
        model_name = f'cnn_{task_id}_{int(time_now())}.safetensors'
        model_path = path_join(user_dir, model_name)

        object.__setattr__(model, '_metadata', {
            'model_type': 'image_cnn',
            'architecture': 'SimpleResNet',
            'model_params': model_params,
            'train_params': train_params,
            'final_accuracy': round(final_acc, 4),
            'final_loss': round(final_loss, 6),
            'total_time': round(time_now() - thread_start, 2),
            'description': f'CNN图像分类模型 | 准确率: {final_acc:.2f}%'
        })

        update_task(training_tasks, task_id,
                   progress=98,
                   message='⏳ 写入磁盘...')
        save_model(model, model_path)
        file_size = getsize(model_path)
        total_time = time_now() - thread_start
        print(f"[CNN] ✅ 模型已保存: {model_path} ({file_size/1024/1024:.1f} MB)")

        # ===== 第6步：写入数据库 =====
        save_model_record(
            user_id=user_id,
            model_name=model_name,
            model_type='cnn',
            file_size=file_size,
            file_path=model_path,
            accuracy=round(final_acc, 4),
            loss=round(final_loss, 6),
            epochs=train_params['epochs']
        )
        print(f"[CNN] ✅ 数据库记录已保存")

        update_task(training_tasks, task_id,
                   status='completed',
                   progress=100,
                   loss=round(final_loss, 6),
                   accuracy=round(final_acc, 4),
                   message=(
                       f'✅ 训练完成！准确率: {final_acc:.2f}% | '
                       f'损失: {final_loss:.4f} | 耗时: {total_time:.0f}s | '
                       f'模型文件: {model_name}'
                   ))
        print(f"========== 🖼️ CNN 训练完成 (总耗时: {total_time:.0f}s) ==========\n")

    except Exception as e:
        err_detail = traceback.format_exc()
        print(f"❌ CNN 训练失败:\n{err_detail}")
        training_tasks[task_id] = {
            'status': 'failed',
            'progress': 0,
            'loss': None,
            'accuracy': None,
            'message': f'❌ 训练失败: {str(e)[:100]}'
        }

# ==================== 文本训练（生成式Transformer，支持MoE/MLA，修复隐性数值损坏问题） ====================
def train_text_model(user_id, task_id, model_params, train_params, training_tasks):
    """生成式文本大模型训练全流程：数据集加载→模型构建→因果语言模型训练→保存→数据库记录"""
    from torch.nn.utils import clip_grad_norm_
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torch.nn.functional as F
    thread_start = time_now()

    try:
        print(f"\n========== 📝 生成式Transformer训练开始 task_id={task_id} ==========")
        print(f"模型参数: {model_params}")
        print(f"训练参数: {train_params}")

        # ===== 第0步：写入初始状态 =====
        training_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'loss': None,
            'ppl': None,
            'message': '🔄 初始化训练线程...'
        }

        # ===== 第1步：加载生成式数据集 =====
        def progress_cb(cur, total, msg):
            if isinstance(total, str):
                scan_pct = 5
            else:
                scan_pct = min(int(cur / max(total, 1) * 15), 14)
            update_task(training_tasks, task_id,
                       progress=scan_pct,
                       message=f'🔍 {msg}')

        update_task(training_tasks, task_id, message='🔍 开始扫描文本训练数据...')

        dataset = TextDataset(
            train_params['data_path'],
            vocab_size=model_params.get('vocab_size', 1000),
            seq_len=model_params.get('max_seq_len', 128),
            progress_callback=progress_cb,
            scan_interval=50,
            pad_token_id=model_params.get('pad_token_id', 0)
        )
        print(f"[Transformer] 数据集统计: 共{len(dataset)}个文本样本")

        # ===== 第2步：构建模型 =====
        update_task(training_tasks, task_id,
                   progress=20,
                   message='🧠 构建生成式Transformer模型...')

        model = TextTransformerModel(
            vocab_size=model_params.get('vocab_size', 1000),
            d_model=model_params.get('d_model', 256),  # 默认改小一点，更稳定
            n_layers=model_params.get('n_layers', 4),
            n_heads=model_params.get('n_heads', 8),
            d_ff=model_params.get('d_ff', 1024),
            max_seq_len=model_params.get('max_seq_len', 128),
            dropout=model_params.get('dropout', 0.1),  # 不要超过0.1，太高容易数值不稳定
            pad_token_id=model_params.get('pad_token_id', 0),
            use_moe=model_params.get('use_moe', False),
            use_mla=model_params.get('use_mla', False),
            moe_experts=model_params.get('moe_experts', 4),
            moe_top_k=model_params.get('moe_top_k', 2),
            mla_dim=model_params.get('mla_dim', 256),
            # 核心修复1：限制aux_loss权重最大0.05，避免MoE辅助损失带崩训练
            aux_loss_weight=min(model_params.get('aux_loss_weight', 0.02), 0.05)
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[Transformer] 模型参数量: {param_count:,}")

        # 可选加速：torch.compile（设环境变量 AITPP_TORCH_COMPILE=1 开启；失败自动回退）
        # 前向走 run_model；保存/评估始终用原始 model，避免编译包装层改变权重键名
        run_model = model
        if os.environ.get('AITPP_TORCH_COMPILE') == '1':
            try:
                run_model = torch.compile(model)
                print("[Transformer] torch.compile 已启用")
            except Exception as e:
                print(f"[Transformer] torch.compile 启用失败，使用 eager 模式: {e}")

        # ===== 第3步：构建DataLoader =====
        update_task(training_tasks, task_id,
                   progress=25,
                   message='📦 准备数据加载器...')

        _workers = _dataloader_workers()
        if _workers > 0 and not _multiproc_loader_ok():
            _workers = 0
            print("[Transformer] 多进程加载不可用，回退单进程")
        loader = DataLoader(
            dataset,
            batch_size=train_params.get('batch_size', 16),  # 默认改小，更稳定
            shuffle=True,
            num_workers=_workers,
            persistent_workers=_workers > 0,
            drop_last=True
        )

        if len(loader) == 0:
            raise ValueError(
                f"数据集样本数({len(dataset)})小于batch_size({train_params['batch_size']})，"
                f"请减小batch_size或增加训练数据"
            )
        print(f"[Transformer] DataLoader: 每轮{len(loader)}个batch")

        # ===== 第4步：训练循环（加全链路数值监控） =====
        # 核心修复2：学习率上限限制为3e-5，大的学习率是梯度爆炸首因
        lr = min(train_params.get('learning_rate', 3e-5), 3e-5)
        opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # 加学习率调度：loss不下降自动降学习率，避免震荡
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0
        max_grad_norm = train_params.get('max_grad_norm', 1.0)  # 梯度裁剪阈值
        # Checkpoint保存路径
        checkpoint_dir = f'models/{user_id}/checkpoints_task_{task_id}'
        makedirs(checkpoint_dir, exist_ok=True)
        last_checkpoint_step = 0
        # 训练中间验证用的小batch
        val_batch = next(iter(loader))  # 取第一个batch做中间验证

        update_task(training_tasks, task_id,
                   progress=26,
                   message='🚀 训练启动...')

        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            batch_num = 0
            # 每个epoch开始先跑一次验证，确保模型输出正常
            model.eval()
            with torch.no_grad():
                try:
                    val_input, val_label = val_batch
                    val_logits = run_model(val_input)
                    # 检测验证输出是否有nan/inf
                    if torch.isnan(val_logits).any() or torch.isinf(val_logits).any():
                        raise RuntimeError("模型验证输出存在nan/inf，前向传播存在数值bug，终止训练")
                except Exception as e:
                    raise RuntimeError(f"Epoch {epoch+1} 验证失败：{str(e)}，请检查模型前向逻辑")
            model.train()  # 切回训练模式

            for batch_idx, (input_ids, labels) in enumerate(loader):
                # 核心修复3：输入参数检测，避免异常token/标签导致训练崩溃
                if input_ids.max() >= model.vocab_size or input_ids.min() < 0:
                    raise ValueError(
                        f"数据集存在异常token！词表大小: {model.vocab_size}, "
                        f"当前batch最大token: {input_ids.max().item()}, 最小token: {input_ids.min().item()}，"
                        f"请检查数据集编码和分词逻辑是否和训练时一致"
                    )
                if labels.max() >= model.vocab_size or labels.min() < -100:
                    raise ValueError(
                        f"数据集标签存在异常值！词表大小: {model.vocab_size}, "
                        f"当前标签最大: {labels.max().item()}, 最小: {labels.min().item()}，"
                        f"标签仅支持0~vocab_size-1和-100（填充标记）"
                    )

                opt.zero_grad()
                
                # 前向传播（支持MoE）
                if model.use_moe:
                    logits, aux_loss = run_model(input_ids, return_aux_loss=True)
                else:
                    logits = run_model(input_ids)
                    aux_loss = torch.tensor(0.0, device=logits.device)

                # 计算损失
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                # 核心修复5：logits数值清洗，避免异常logits导致损失爆炸
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                cls_loss = criterion(logits, labels)
                # 主损失清洗
                cls_loss = torch.nan_to_num(cls_loss, nan=1e-8, posinf=1e-8, neginf=1e-8)
                loss = cls_loss + aux_loss

                # 核心修复6：梯度爆炸检测+裁剪
                loss.backward()
                # 先检测梯度是否有nan/inf，有立刻终止，避免污染参数
                has_nan_grad = False
                has_inf_grad = False
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            has_nan_grad = True
                        if torch.isinf(p.grad).any():
                            has_inf_grad = True
                if has_nan_grad or has_inf_grad:
                    raise RuntimeError(
                        f"第{epoch+1}轮第{batch_idx+1}步出现梯度nan/inf，已终止训练避免模型损坏，"
                        f"建议降低学习率、增大梯度裁剪阈值、检查数据集是否有异常"
                    )
                # 执行梯度裁剪
                total_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

                total_loss += loss.item()
                batch_count += 1
                batch_num += 1

                # 核心修复7：每100个batch检测一次模型参数，提前发现隐性损坏
                if batch_count % 100 == 0:
                    param_has_nan = any(torch.isnan(p).any() for p in model.parameters())
                    param_has_inf = any(torch.isinf(p).any() for p in model.parameters())
                    if param_has_nan or param_has_inf:
                        raise RuntimeError(
                            f"第{epoch+1}轮第{batch_idx+1}步检测到模型参数出现nan/inf，"
                            f"训练过程数值不稳定，已终止训练"
                        )

                # 进度更新
                progress = 26 + int((batch_count / max(total_batches, 1)) * 69)
                progress = min(progress, 95)
                current_loss = loss.item()
                current_ppl = round(math.exp(min(current_loss, 100)), 2)  # 防溢出

                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(current_loss, 6),
                    'ppl': current_ppl,
                    'message': (
                        f'Epoch {epoch+1}/{train_params["epochs"]} '
                        f'| Batch {batch_idx+1}/{len(loader)} '
                        f'| Loss: {current_loss:.4f} | PPL: {current_ppl} | GradNorm: {total_norm:.2f}'
                    )
                })

                # 核心修复8：每500步自动保存Checkpoint，避免训练崩了丢进度
                if batch_count - last_checkpoint_step >= 500:
                    checkpoint_path = path_join(checkpoint_dir, f'checkpoint_epoch{epoch+1}_step{batch_idx+1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': current_loss,
                        'ppl': current_ppl,
                        'has_nan': False,
                        'has_inf': False
                    }, checkpoint_path)
                    last_checkpoint_step = batch_count
                    print(f"[Transformer] 已保存Checkpoint: {checkpoint_path}")

            # Epoch结束统计
            epoch_time = time_now() - epoch_start
            avg_loss = total_loss / len(loader)
            avg_ppl = round(math.exp(min(avg_loss, 100)), 2)
            print(f"[Transformer] Epoch {epoch+1}/{train_params['epochs']} | "
                  f'Loss: {avg_loss:.4f} | PPL: {avg_ppl} | '
                  f'耗时: {epoch_time:.1f}s | 平均梯度范数: {total_norm:.2f}')
            
            # 更新学习率调度
            lr_scheduler.step(avg_loss)
            print(f"[Transformer] 当前学习率: {opt.param_groups[0]['lr']:.2e}")

        # 最终指标
        final_loss = total_loss / len(loader)
        final_ppl = round(math.exp(min(final_loss, 100)), 2)

        # ===== 第5步：保存模型（加最终参数检测） =====
        update_task(training_tasks, task_id,
                   progress=96,
                   message='💾 正在保存模型...')

        # 保存前全量检测参数
        has_nan = any(torch.isnan(p).any() for p in model.parameters())
        has_inf = any(torch.isinf(p).any() for p in model.parameters())
        if has_nan or has_inf:
            raise RuntimeError("模型训练后参数存在nan/inf，保存失败，请检查训练过程是否出现梯度爆炸")

        user_dir = f'models/{user_id}'
        makedirs(user_dir, exist_ok=True)
        model_name = f'text_gen_{task_id}_{int(time_now())}.safetensors'
        model_path = path_join(user_dir, model_name)

        object.__setattr__(model, '_metadata', {
            'model_type': 'text_generation',
            'architecture': 'Transformer-Decoder（生成式）',
            'model_params': model_params,
            'train_params': train_params,
            'final_ppl': final_ppl,
            'final_loss': round(final_loss, 6),
            'total_time': round(time_now() - thread_start, 2),
            'description': f'生成式文本大模型 | 困惑度: {final_ppl} | MoE: {model.use_moe} | 参数量: {param_count:,}'
        })

        update_task(training_tasks, task_id,
                   progress=98,
                   message='⏳ 写入磁盘...')
        save_model(model, model_path)
        file_size = getsize(model_path)
        total_time = time_now() - thread_start
        print(f"[Transformer] ✅ 模型已保存: {model_path} ({file_size/1024/1024:.1f} MB)")

        # ===== 第6步：写入数据库 =====
        save_model_record(
            user_id=user_id,
            model_name=model_name,
            model_type='text',
            file_size=file_size,
            file_path=model_path,
            accuracy=final_ppl,
            loss=round(final_loss, 6),
            epochs=train_params['epochs']
        )
        print(f"[Transformer] ✅ 数据库记录已保存")

        update_task(training_tasks, task_id,
                   status='completed',
                   progress=100,
                   loss=round(final_loss, 6),
                   ppl=final_ppl,
                   message=(
                       f'✅ 训练完成！困惑度: {final_ppl} | '
                       f'损失: {final_loss:.4f} | 耗时: {total_time:.0f}s | '
                       f'模型文件: {model_name}'
                   ))
        print(f"========== 📝 生成式Transformer训练完成 (总耗时: {total_time:.0f}s) ==========\n")

    except Exception as e:
        err_detail = traceback.format_exc()
        print(f"❌ Transformer 训练失败:\n{err_detail}")
        # 失败时提示Checkpoint续训
        error_msg = str(e)[:100]
        if 'gradient nan/inf' in err_detail or '梯度爆炸' in err_detail or '验证输出存在nan/inf' in err_detail:
            error_msg = "模型前向/训练出现数值异常，建议降低学习率、检查模型前向逻辑。已保留最近Checkpoint，可续训"
        elif '异常token' in err_detail:
            error_msg = "数据集存在异常token，请检查数据编码和分词逻辑是否和训练一致"
        training_tasks[task_id] = {
            'status': 'failed',
            'progress': 0,
            'loss': None,
            'ppl': None,
            'message': f'❌ 训练失败: {error_msg}'
        }