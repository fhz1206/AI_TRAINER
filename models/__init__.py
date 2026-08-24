"""
models — 模型注册表

按类型名构建模型（Web 端积木式选择的基础）：
    from models import available_models, build_model
    net = build_model('image_vit', image_size=32, num_classes=5, attention_type='flash')

内置类型：
    image_cnn        CNN 图像分类（SimpleResNet）
    image_vit        ViT 图像分类（可插拔注意力）
    text_generation  Decoder-only 文本生成（可插拔注意力 + 可选 MoE）
扩展：新模型在自身模块里调用 register_model('名字') 注册即可进入训练页选项。
"""

_MODEL_REGISTRY = {}


def register_model(name):
    """注册装饰器：把模型类挂入全局注册表"""
    def deco(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return deco


def available_models():
    return sorted(_MODEL_REGISTRY)


def get_model_class(name):
    cls = _MODEL_REGISTRY.get((name or '').lower())
    if cls is None:
        raise ValueError(f"未知模型类型: {name}，可选: {available_models()}")
    return cls


def build_model(name, **kwargs):
    """按注册名实例化模型；参数直接透传给对应类的构造函数"""
    return get_model_class(name)(**kwargs)


from .vision import ConvBlock, SimpleResNet, ViTModel            # noqa: E402,F401
from .text import TextTransformerModel                           # noqa: E402,F401

register_model('image_cnn')(SimpleResNet)
register_model('image_vit')(ViTModel)
register_model('text_generation')(TextTransformerModel)

# 扩散 / 多模态为可选组件，缺失时跳过注册（不影响文本与图像分类）
try:                                                              # noqa: E402
    from .diffusion import DiffusionModel, DiffusionEditModel     # noqa: F401
    register_model('image_diffusion')(DiffusionModel)
    register_model('image_edit_diffusion')(DiffusionEditModel)
except ImportError:
    pass

try:                                                              # noqa: E402
    from .multimodal import MultiModalSingleStream                # noqa: F401
    register_model('multimodal_stream')(MultiModalSingleStream)
except ImportError:
    pass
