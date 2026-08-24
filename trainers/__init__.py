"""
trainers — 训练器包

按训练类型分发到对应实现：
    from trainers import start_training_thread
    start_training_thread('image_vit', user_id, model_params, train_params)

训练类型 → 实现函数：
    image_cnn / image_vit          → image_cls.train_image_classification
    text_generation                → text_gen.train_text_model
    image_diffusion                → diffusion.train_diffusion
    image_edit_diffusion           → diffusion.train_diffusion_edit
    multimodal_stream              → multimodal.train_multimodal

每个训练器在独立线程中运行（Flask 请求线程立即返回 task_id，前端轮询进度）。
"""
import traceback
from threading import Thread

from state import training_tasks
from .common import update_task, build_loader, maybe_compile, fail_task  # noqa: F401
from .image_cls import train_image_classification
from .text_gen import train_text_model
from .diffusion import train_diffusion, train_diffusion_edit
from .multimodal import train_multimodal

TRAIN_FUNCTIONS = {
    'image_cnn': lambda uid, tid, mp, tp, tasks:
        train_image_classification('image_cnn', uid, tid, mp, tp, tasks),
    'image_vit': lambda uid, tid, mp, tp, tasks:
        train_image_classification('image_vit', uid, tid, mp, tp, tasks),
    'text_generation': train_text_model,
    'image_diffusion': train_diffusion,
    'image_edit_diffusion': train_diffusion_edit,
    'multimodal_stream': train_multimodal,
}

# 前端分区 → 可选架构（积木式选择的数据源）
TRAIN_TYPES = {
    'llm': ['text_generation'],
    'image': ['image_cnn', 'image_vit', 'image_diffusion', 'image_edit_diffusion'],
    'multimodal': ['multimodal_stream'],
}


def start_training_thread(train_type, user_id, model_params,
                          train_params, training_tasks=None):
    """
    校验类型并启动训练线程；返回 (成功?, 任务ID或错误信息)。
    training_tasks 缺省用全局状态表。
    """
    from uuid import uuid4
    tasks = training_tasks if training_tasks is not None else training_tasks_global()
    fn = TRAIN_FUNCTIONS.get((train_type or '').lower())
    if fn is None:
        return False, f'未知训练类型: {train_type}，可选: {sorted(TRAIN_FUNCTIONS)}'
    task_id = str(uuid4())[:8]

    def _run():
        try:
            fn(user_id, task_id, model_params, train_params, tasks)
        except Exception as e:  # 双保险：训练器内部已兜底，此处防御分发层异常
            print(f"❌ 训练线程异常:\n{traceback.format_exc()}")
            tasks[task_id] = {
                'status': 'failed', 'progress': 0, 'loss': None,
                'accuracy': None, 'message': f'❌ 训练失败: {str(e)[:100]}'
            }

    Thread(target=_run, daemon=True, name=f'train-{task_id}').start()
    return True, task_id


def training_tasks_global():
    return training_tasks
