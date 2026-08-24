"""
trainers.common — 训练器公共设施

包括：任务状态更新、DataLoader 多进程构建（含受限环境自动回退）、
torch.compile 可选包装。所有训练器共用。
"""
import os

import torch
from torch.utils.data import DataLoader


def update_task(tasks, task_id, **kwargs):
    """线程安全地更新全局训练任务状态"""
    if task_id in tasks:
        tasks[task_id].update(kwargs)


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


def build_loader(dataset, batch_size, tag='', shuffle=True):
    """构建多进程 DataLoader；受限环境自动回退单进程"""
    workers = _dataloader_workers()
    if workers > 0 and not _multiproc_loader_ok():
        workers = 0
        print(f"[{tag}] 多进程加载不可用，回退单进程")
    kwargs = dict(num_workers=workers,
                  persistent_workers=workers > 0,
                  drop_last=True)
    if workers > 0:
        kwargs['pin_memory'] = True
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def maybe_compile(model, tag=''):
    """可选 torch.compile 包装（AITPP_TORCH_COMPILE=1 开启），失败回退原模型"""
    run_model = model
    if os.environ.get('AITPP_TORCH_COMPILE') == '1':
        try:
            run_model = torch.compile(model)
            print(f"[{tag}] torch.compile 已启用")
        except Exception as e:
            print(f"[{tag}] torch.compile 启用失败，使用 eager 模式: {e}")
    return run_model


def fail_task(training_tasks, task_id, e, tb):
    """统一的失败状态写入"""
    print(f"❌ 训练失败:\n{tb}")
    training_tasks[task_id] = {
        'status': 'failed',
        'progress': 0,
        'loss': None,
        'accuracy': None,
        'message': f'❌ 训练失败: {str(e)[:100]}'
    }
