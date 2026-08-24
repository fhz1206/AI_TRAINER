"""
trainer.py — 兼容垫片

训练实现已迁移至 trainers 包（按架构拆分：图像分类 / 文本生成 / 扩散 / 多模态）。
本文件仅保留旧导入路径：
    from trainer import train_image_model, train_text_model
新代码请使用：
    from trainers import start_training_thread, TRAIN_FUNCTIONS, TRAIN_TYPES
"""
from trainers.text_gen import train_text_model  # noqa: F401
from trainers.common import update_task  # noqa: F401
from trainers.image_cls import train_image_classification  # noqa: F401


def train_image_model(user_id, task_id, model_params, train_params, training_tasks):
    """旧接口兼容：等价于 CNN 图像分类训练（新代码请用 trainers.start_training_thread）"""
    return train_image_classification(
        'image_cnn', user_id, task_id, model_params, train_params, training_tasks)
