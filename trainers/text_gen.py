"""
trainers.text_gen — 生成式文本 Transformer 训练器

迁移自原 trainer.train_text_model，全部数值防护逻辑保留：
学习率上限、梯度 nan/inf 检测与裁剪、参数损坏巡检、Checkpoint 续训提示。
新增：attention_type 积木选择（full/flash/linear），MoE 使用修复版实现。
"""
import traceback
from os import makedirs
from os.path import join as path_join, getsize
from time import time as time_now

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

from model import TextDataset
from model_zoo.text import TextTransformerModel
from model_io import save_model
from database import save_model_record

from .common import update_task, build_loader, maybe_compile, fail_task


def train_text_model(user_id, task_id, model_params, train_params, training_tasks):
    """生成式语言模型全流程：数据集 → 构建 → 因果 LM 训练 → 保存 → 记录数据库"""
    thread_start = time_now()
    try:
        print(f"\n========== 📝 生成式Transformer训练开始 task_id={task_id} ==========")
        print(f"模型参数: {model_params}")
        print(f"训练参数: {train_params}")

        training_tasks[task_id] = {
            'status': 'running', 'progress': 0, 'loss': None,
            'ppl': None, 'message': '🔄 初始化训练线程...'
        }

        # ---- 数据集 ----
        def progress_cb(cur, total, msg):
            pct = 5 if isinstance(total, str) else min(int(cur / max(total, 1) * 15), 14)
            update_task(training_tasks, task_id, progress=pct, message=f'🔍 {msg}')

        dataset = TextDataset(
            train_params['data_path'],
            vocab_size=model_params.get('vocab_size', 1000),
            seq_len=model_params.get('max_seq_len', 128),
            progress_callback=progress_cb,
            scan_interval=50,
            pad_token_id=model_params.get('pad_token_id', 0),
        )
        print(f"[Transformer] 数据集统计: 共{len(dataset)}个文本样本")

        # ---- 模型 ----
        update_task(training_tasks, task_id, progress=20,
                    message='🧠 构建生成式Transformer模型...')
        model = TextTransformerModel(
            vocab_size=model_params.get('vocab_size', 1000),
            d_model=model_params.get('d_model', 256),
            n_layers=model_params.get('n_layers', 4),
            n_heads=model_params.get('n_heads', 8),
            d_ff=model_params.get('d_ff', 1024),
            max_seq_len=model_params.get('max_seq_len', 128),
            dropout=model_params.get('dropout', 0.1),
            pad_token_id=model_params.get('pad_token_id', 0),
            use_moe=model_params.get('use_moe', False),
            use_mla=model_params.get('use_mla', False),
            moe_experts=model_params.get('moe_experts', 4),
            moe_top_k=model_params.get('moe_top_k', 2),
            mla_dim=model_params.get('mla_dim', 256),
            aux_loss_weight=min(model_params.get('aux_loss_weight', 0.02), 0.05),
            attention_type=model_params.get('attention_type', 'flash'),
            attention_plan=model_params.get('attention_plan'),
        )
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[Transformer] 模型参数量: {param_count:,}")
        run_model = maybe_compile(model, 'Transformer')

        # ---- 数据加载器 ----
        update_task(training_tasks, task_id, progress=25, message='📦 准备数据加载器...')
        loader = build_loader(dataset, train_params.get('batch_size', 16),
                              tag='Transformer')
        if len(loader) == 0:
            raise ValueError(
                f"数据集样本数({len(dataset)})小于batch_size({train_params['batch_size']})，"
                f"请减小batch_size或增加训练数据")

        # ---- 训练准备 ----
        lr = min(train_params.get('learning_rate', 3e-5), 3e-5)   # 上限防梯度爆炸
        opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        total_batches = len(loader) * train_params['epochs']
        batch_count = 0
        max_grad_norm = train_params.get('max_grad_norm', 1.0)
        checkpoint_dir = f'models/{user_id}/checkpoints_task_{task_id}'
        makedirs(checkpoint_dir, exist_ok=True)
        last_checkpoint_step = 0
        val_batch = next(iter(loader))

        update_task(training_tasks, task_id, progress=26, message='🚀 训练启动...')

        for epoch in range(train_params['epochs']):
            epoch_start = time_now()
            total_loss = 0.0
            batch_num = 0
            model.eval()
            with torch.no_grad():
                try:
                    val_input, _ = val_batch
                    val_logits = run_model(val_input)
                    if torch.isnan(val_logits).any() or torch.isinf(val_logits).any():
                        raise RuntimeError("模型验证输出存在nan/inf")
                except Exception as e:
                    raise RuntimeError(f"Epoch {epoch+1} 验证失败：{str(e)}") from e
            model.train()

            for batch_idx, (input_ids, labels) in enumerate(loader):
                if input_ids.max() >= model.vocab_size or input_ids.min() < 0:
                    raise ValueError(
                        f"数据集存在异常token！词表大小: {model.vocab_size}, "
                        f"当前batch最大token: {input_ids.max().item()}, "
                        f"最小token: {input_ids.min().item()}")
                if labels.max() >= model.vocab_size or labels.min() < -100:
                    raise ValueError(
                        f"数据集标签存在异常值！当前标签最大: {labels.max().item()}, "
                        f"最小: {labels.min().item()}（仅支持0~vocab-1和-100）")

                opt.zero_grad()
                if model.use_moe:
                    logits, aux_loss = run_model(input_ids, return_aux_loss=True)
                else:
                    logits = run_model(input_ids)
                    aux_loss = torch.tensor(0.0, device=logits.device)

                logits = logits.view(-1, logits.size(-1))
                labels_v = labels.view(-1)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                cls_loss = criterion(logits, labels_v)
                cls_loss = torch.nan_to_num(cls_loss, nan=1e-8, posinf=1e-8, neginf=1e-8)
                loss = cls_loss + aux_loss

                loss.backward()
                has_bad_grad = any(
                    p.grad is not None and (
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in model.parameters())
                if has_bad_grad:
                    raise RuntimeError(
                        f"第{epoch+1}轮第{batch_idx+1}步出现梯度nan/inf，已终止避免模型损坏；"
                        f"建议降低学习率或检查数据集")
                clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()

                total_loss += loss.item()
                batch_count += 1
                batch_num += 1

                if batch_count % 100 == 0:
                    bad_param = any(torch.isnan(p).any() or torch.isinf(p).any()
                                    for p in model.parameters())
                    if bad_param:
                        raise RuntimeError("检测到模型参数nan/inf，提前终止保护产物")

                progress = min(26 + int((batch_count / max(total_batches, 1)) * 69), 95)
                ppl = float(torch.exp(loss.detach().clamp(max=20)))
                training_tasks[task_id].update({
                    'progress': progress,
                    'loss': round(loss.item(), 6),
                    'ppl': round(ppl, 3),
                    'message': (f'Epoch {epoch+1}/{train_params["epochs"]} '
                                f'| Batch {batch_idx+1}/{len(loader)} '
                                f'| Loss: {loss.item():.4f} | PPL: {ppl:.1f}')
                })

                # 每500步存 checkpoint，崩溃可续训
                if batch_count - last_checkpoint_step >= 500:
                    torch.save({
                        'epoch': epoch, 'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                    }, path_join(checkpoint_dir,
                                 f'checkpoint_e{epoch+1}_s{batch_count}.pt'))
                    last_checkpoint_step = batch_count
                    print(f"[Transformer] 已保存Checkpoint: e{epoch+1}_s{batch_count}")

            avg_loss = total_loss / max(len(loader), 1)
            lr_scheduler.step(avg_loss)
            print(f"[Transformer] Epoch {epoch+1}/{train_params['epochs']} | "
                  f'Loss: {avg_loss:.4f} | 耗时: {time_now()-epoch_start:.1f}s')

        final_loss = total_loss / max(len(loader), 1)
        final_ppl = round(float(torch.exp(torch.tensor(min(final_loss, 20)))), 2)

        # ---- 保存 ----
        update_task(training_tasks, task_id, progress=96, message='💾 正在保存模型...')
        user_dir = f'models/{user_id}'
        makedirs(user_dir, exist_ok=True)
        model_name = f'text_gen_{task_id}_{int(time_now())}.safetensors'
        model_path = path_join(user_dir, model_name)

        object.__setattr__(model, '_metadata', {
            'model_type': 'text_generation',
            'architecture': f'Transformer-Decoder({model.attention_type} attention)',
            'model_params': model_params,
            'train_params': train_params,
            'final_ppl': final_ppl,
            'final_loss': round(final_loss, 6),
            'total_time': round(time_now() - thread_start, 2),
            'description': (f'生成式文本大模型 | 困惑度: {final_ppl} | '
                            f'注意力: {model.attention_type} | MoE: {model.use_moe}')
        })

        update_task(training_tasks, task_id, progress=98, message='⏳ 写入磁盘...')
        save_model(model, model_path)
        file_size = getsize(model_path)
        total_time = time_now() - thread_start

        save_model_record(user_id=user_id, model_name=model_name, model_type='text',
                          file_size=file_size, file_path=model_path,
                          accuracy=final_ppl, loss=round(final_loss, 6),
                          epochs=train_params['epochs'])

        update_task(training_tasks, task_id,
                    status='completed', progress=100,
                    loss=round(final_loss, 6), ppl=final_ppl,
                    message=(f'✅ 训练完成！困惑度: {final_ppl} | '
                             f'损失: {final_loss:.4f} | 耗时: {total_time:.0f}s | '
                             f'模型文件: {model_name}'))
        print(f"========== 📝 训练完成 (总耗时: {total_time:.0f}s) ==========\n")

    except Exception as e:
        err_detail = traceback.format_exc()
        error_msg = str(e)[:100]
        if 'nan/inf' in err_detail or '梯度爆炸' in err_detail:
            error_msg = "数值异常，建议降低学习率。已保留最近Checkpoint可续训"
        elif '异常token' in err_detail:
            error_msg = "数据集存在异常token，请检查分词逻辑"
        fail_task(training_tasks, task_id, Exception(error_msg), err_detail)
