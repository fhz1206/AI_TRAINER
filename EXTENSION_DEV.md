# 扩展开发指南

> 本文档面向希望为本平台开发第三方扩展的开发者，介绍扩展系统的架构、开发流程和最佳实践。

---

## 目录

- [扩展系统概述](#扩展系统概述)
- [扩展目录结构](#扩展目录结构)
- [manifest.json 详解](#manifestjson-详解)
- [支持的扩展类型](#支持的扩展类型)
- [钩子（Hooks）系统](#钩子hooks系统)
- [前端扩展](#前端扩展)
- [后端扩展](#后端扩展)
- [完整示例：图片数据增强扩展](#完整示例图片数据增强扩展)
- [调试与测试](#调试与测试)
- [发布与分享](#发布与分享)
- [常见问题](#常见问题)

---

## 扩展系统概述

本平台采用**插件目录扫描式**扩展架构，扩展以独立目录形式存放在 `extensions/` 文件夹下，系统启动时自动扫描加载。每个扩展包含一个 `manifest.json` 声明文件和 Python 实现代码。

### 核心特性

- **即插即用**：将扩展目录放入 `extensions/` 即可，无需修改核心代码
- **独立前端资源**：每个扩展可携带独立的 CSS、JS 和模板文件
- **权限控制**：扩展可声明所需权限，管理员可启用/禁用
- **版本管理**：扩展可声明依赖的 Python 包版本
- **热加载**：支持在运行时重新加载扩展（开发模式下）

---

## 扩展目录结构

```
extensions/
├── __init__.py              # 扩展包初始化
├── example_extension/       # 一个扩展的完整示例
│   ├── manifest.json        # 扩展元数据声明（必需）
│   ├── __init__.py           # 扩展入口（必需）
│   ├── backend.py            # 后端逻辑实现
│   ├── models.py             # 自定义模型定义（可选）
│   ├── trainers.py           # 自定义训练器（可选）
│   ├── static/               # 前端资源（可选）
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── script.js
│   └── templates/            # 模板文件（可选）
│       └── config.html
├── another_extension/
│   └── manifest.json
│   └── __init__.py
```

---

## manifest.json 详解

每个扩展必须包含 `manifest.json` 文件，定义扩展的元数据和接口声明。

```json
{
  "name": "图片数据增强",
  "version": "1.0.0",
  "author": "fhz",
  "description": "为训练数据添加随机裁剪、旋转、色彩抖动等增强",
  "type": "preprocessor",
  "hooks": ["before_train", "after_upload"],
  "routes": [
    {"method": "POST", "path": "/api/ext/augment/preview", "handler": "preview"},
    {"method": "GET",  "path": "/api/ext/augment/config", "handler": "get_config"}
  ],
  "requires": ["imgaug>=0.4.0"],
  "admin_panel": true,
  "permissions": ["read_uploads", "write_models"],
  "frontend": {
    "entry": "static/js/script.js",
    "styles": ["static/css/style.css"],
    "templates": ["templates/config.html"]
  }
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | 扩展名称，显示在管理面板 |
| `version` | string | ✅ | 语义化版本号 |
| `author` | string | ❌ | 开发者名称 |
| `description` | string | ✅ | 功能描述 |
| `type` | string | ✅ | 扩展类型（见下文） |
| `hooks` | array | ❌ | 注册的事件钩子列表 |
| `routes` | array | ❌ | 自定义 API 路由 |
| `requires` | array | ❌ | 依赖的 Python 包 |
| `admin_panel` | bool | ❌ | 是否在管理面板显示配置页 |
| `permissions` | array | ❌ | 所需权限列表 |
| `frontend` | object | ❌ | 前端资源声明 |

---

## 支持的扩展类型

| type | 说明 | 示例 |
|------|------|------|
| `model` | 新增模型架构 | ResNet、RNN、GAN 等 |
| `trainer` | 新增训练算法 | 强化学习、对抗训练、联邦学习 |
| `preprocessor` | 数据预处理/增强 | 图片裁剪、旋转、噪声添加 |
| `visualizer` | 可视化工具 | 训练曲线、模型结构图、特征图 |
| `test_framework` | 测试框架扩展 | 自动化测试、性能基准测试 |
| `auth_provider` | 第三方登录 | OAuth、LDAP、企业微信登录 |
| `ui_theme` | 界面主题 | 自定义皮肤、布局 |
| `export` | 模型导出 | ONNX、TensorRT、TFLite |

---

## 钩子（Hooks）系统

扩展可以通过注册钩子函数，在平台的关键流程中插入自定义逻辑。

### 可用钩子

| 钩子名称 | 触发时机 | 回调参数 |
|----------|----------|----------|
| `before_request` | 每个 HTTP 请求前 | `(request)` |
| `after_request` | 每个 HTTP 响应后 | `(response)` |
| `before_upload` | 文件上传前 | `(file, user_id, train_type)` |
| `after_upload` | 文件上传完成 | `(file_record, user_id)` |
| `before_train` | 训练开始前 | `(task_id, user_id, params)` |
| `after_epoch` | 每轮训练结束 | `(epoch, loss, accuracy, model)` |
| `after_train` | 训练完成 | `(task_id, model_path, metrics)` |
| `before_test` | 测试开始前 | `(task_id, framework, code)` |
| `after_test` | 测试完成 | `(task_id, results)` |
| `on_error` | 发生错误时 | `(error, context)` |

### 注册钩子

```python
# extensions/my_extension/__init__.py
from core.hooks import register_hook

def on_after_epoch(epoch, loss, accuracy, model):
    """每轮训练后记录到自定义日志"""
    log_to_my_service(epoch, loss, accuracy)
    # 可以修改模型参数
    return model

register_hook('after_epoch', on_after_epoch)
```

---

## 前端扩展

### 添加自定义页面

扩展可以通过 `frontend.templates` 声明自定义模板页面，这些页面会注入到主平台的指定位置。

```json
{
  "frontend": {
    "templates": ["templates/config.html"],
    "inject_to": ["admin_panel", "training_panel"]
  }
}
```

### 添加自定义 API

扩展通过 `routes` 字段声明自定义 API 端点，这些端点会自动注册到 Flask 应用。

```python
# extensions/my_extension/backend.py
from flask import jsonify, request

def preview(data):
    """预览增强效果"""
    import cv2
    import numpy as np
    img_path = data.get('path')
    img = cv2.imread(img_path)
    # 应用增强
    augmented = apply_augmentation(img)
    return jsonify({'status': 'success', 'preview': augmented.tolist()})

def get_config():
    """获取扩展配置"""
    return jsonify({
        'rotation_range': 30,
        'brightness_range': 0.2,
        'crop_ratio': 0.1
    })
```

### 1. 在训练面板添加选项卡

扩展可以在训练面板的图片/文字训练标签页旁边添加自定义选项卡：

```javascript
// extensions/my_extension/static/js/script.js
// 在训练面板添加自定义选项卡
document.addEventListener('DOMContentLoaded', function() {
    const tabsContainer = document.querySelector('.tabs-container');
    if (tabsContainer) {
        const btn = document.createElement('button');
        btn.className = 'tab-btn';
        btn.dataset.type = 'my_extension';
        btn.innerHTML = '<span>🔧</span> 我的扩展';
        btn.onclick = () => switchType('my_extension');
        tabsContainer.appendChild(btn);
    }
});
```

### 2. 在管理员面板添加配置页

扩展可以有自己的管理员配置页面，通过 `admin_panel: true` 启用。

---

## 后端扩展

### 扩展入口文件

`__init__.py` 是扩展的入口，必须包含 `init_extension(app)` 函数。

```python
# extensions/my_extension/__init__.py
from flask import Blueprint

def init_extension(app):
    """扩展初始化入口，app 是 Flask 应用实例"""
    # 注册蓝图
    bp = Blueprint('ext_my_ext', __name__, url_prefix='/api/ext/my_ext')
    from . import backend
    bp.add_url_rule('/preview', view_func=backend.preview, methods=['POST'])
    bp.add_url_rule('/config', view_func=backend.get_config, methods=['GET'])
    app.register_blueprint(bp)
    
    # 注册钩子
    from core.hooks import register_hook
    register_hook('after_epoch', backend.on_after_epoch)
    
    print(f"[扩展] 已加载: 我的扩展")
```

### 使用核心模块

扩展可以直接导入并使用平台的核心模块：

```python
from model import SimpleResNet, TextTransformerModel  # 复用模型
from trainer import train_image_model, train_text_model  # 复用训练器
from database import get_db, save_model_record  # 操作数据库
from state import training_tasks  # 访问训练状态
```

### 自定义模型

扩展可以定义新的模型架构，并注册到平台中供用户选择：

```python
# extensions/rnn_model/models.py
import torch.nn as nn
from model import register_model

@register_model('rnn', 'RNN 文本分类模型')
class RNNModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=2, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.RNN(d_model, d_model, n_layers, batch_first=True)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x
```

---

## 完整示例：图片数据增强扩展

下面是一个完整的扩展示例，实现训练数据的图片增强功能。

### 目录结构

```
extensions/image_augment/
├── manifest.json
├── __init__.py
├── backend.py
├── static/
│   └── js/
│       └── augment.js
└── templates/
    └── config.html
```

### manifest.json

```json
{
  "name": "图片数据增强",
  "version": "1.0.0",
  "author": "fhz",
  "description": "为训练数据添加随机裁剪、旋转、色彩抖动等增强",
  "type": "preprocessor",
  "hooks": ["before_train"],
  "routes": [
    {"method": "POST", "path": "/api/ext/augment/preview", "handler": "preview"},
    {"method": "GET",  "path": "/api/ext/augment/config", "handler": "get_config"}
  ],
  "requires": ["imgaug>=0.4.0"],
  "admin_panel": true,
  "frontend": {
    "entry": "static/js/augment.js",
    "templates": ["templates/config.html"]
  }
}
```

### __init__.py

```python
from flask import Blueprint, jsonify

def init_extension(app):
    bp = Blueprint('ext_augment', __name__, url_prefix='/api/ext/augment')
    from . import backend
    bp.add_url_rule('/preview', view_func=backend.preview, methods=['POST'])
    bp.add_url_rule('/config', view_func=backend.get_config, methods=['GET'])
    app.register_blueprint(bp)
    from core.hooks import register_hook
    register_hook('before_train', backend.on_before_train)
    print("[扩展] 已加载: 图片数据增强")
```

### backend.py

```python
import imgaug.augmenters as iaa
import cv2
import numpy as np
from flask import jsonify, request

# 默认增强配置
_aug_config = {
    'rotation_range': 30,
    'brightness_range': 0.2,
    'crop_ratio': 0.1,
    'flip_prob': 0.5,
}

def get_augmenter(config=None):
    if config is None:
        config = _aug_config
    return iaa.Sequential([
        iaa.Affine(rotate=(-config['rotation_range'], config['rotation_range'])),
        iaa.Multiply((1.0 - config['brightness_range'], 1.0 + config['brightness_range'])),
        iaa.Crop(percent=(0, config['crop_ratio'])),
        iaa.Fliplr(config['flip_prob']),
    ])

def preview():
    data = request.json
    img_path = data.get('path')
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({'status': 'error', 'message': '图片读取失败'})
    aug = get_augmenter()
    augmented = aug(image=img)
    _, buffer = cv2.imencode('.jpg', augmented)
    return jsonify({
        'status': 'success',
        'preview': buffer.tobytes().hex()
    })

def get_config():
    return jsonify({'status': 'success', 'config': _aug_config})

def on_before_train(task_id, user_id, params):
    """在训练前对数据集进行增强"""
    from config import IMAGE_EXTENSIONS
    import os
    data_path = params.get('train_params', {}).get('data_path', '')
    if not data_path or not os.path.isdir(data_path):
        return
    aug = get_augmenter()
    for root, _, files in os.walk(data_path):
        for f in files:
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                img_path = os.path.join(root, f)
                img = cv2.imread(img_path)
                if img is not None:
                    augmented = aug(image=img)
                    cv2.imwrite(img_path, augmented)
    print(f"[增强] 数据集增强完成: {data_path}")
```

### static/js/augment.js

```javascript
// 在训练面板添加增强配置按钮
document.addEventListener('DOMContentLoaded', function() {
    // 在训练参数区域添加增强配置
    const paramsCard = document.getElementById('imageParams');
    if (paramsCard) {
        const section = document.createElement('div');
        section.className = 'card';
        section.innerHTML = `
            <div class="card-header">
                <span class="icon">✨</span>
                <h3>图片增强配置</h3>
            </div>
            <div class="param-grid-3">
                <div class="form-group">
                    <label>旋转范围</label>
                    <input type="number" id="aug_rotation" value="30" min="0" max="180">
                </div>
                <div class="form-group">
                    <label>亮度范围</label>
                    <input type="number" id="aug_brightness" value="0.2" min="0" max="1" step="0.1">
                </div>
                <div class="form-group">
                    <label>裁剪比例</label>
                    <input type="number" id="aug_crop" value="0.1" min="0" max="0.5" step="0.05">
                </div>
            </div>
        `;
        paramsCard.parentNode.insertBefore(section, paramsCard.nextSibling);
    }
});
```

---

## 调试与测试

### 开发模式

1. 将扩展目录放入 `extensions/` 文件夹
2. 在 `config.py` 中设置 `DEBUG = True`
3. 启动服务器，查看控制台输出确认扩展已加载
4. 修改扩展代码后，重启服务器即可生效

### 测试扩展

```python
# 使用 pytest 测试扩展功能
def test_extension_loaded():
    from extensions.image_augment import init_extension
    assert init_extension is not None

def test_manifest_valid():
    import json
    with open('extensions/image_augment/manifest.json') as f:
        manifest = json.load(f)
    assert 'name' in manifest
    assert 'version' in manifest
```

### 常见错误排查

| 错误 | 可能原因 |
|------|----------|
| 扩展未加载 | 检查 `extensions/__init__.py` 是否存在 |
| manifest.json 解析失败 | 检查 JSON 格式是否正确 |
| 路由冲突 | 扩展路由前缀必须为 `/api/ext/` |
| 依赖缺失 | 检查 `requires` 中的包是否已安装 |
| 前端资源未加载 | 检查 `static/` 路径是否正确 |

---

## 发布与分享

1. 将扩展目录打包为 ZIP 文件
2. 在 README 中说明扩展的安装方式
3. 用户将 ZIP 解压到 `extensions/` 目录即可使用

### 分享清单

- 扩展名称和版本号
- 功能描述和截图
- 安装步骤
- 依赖的 Python 包
- 兼容的平台版本
- 联系方式

---

## 常见问题

### Q: 扩展可以访问数据库吗？

A: 可以。扩展可以直接导入 `database` 模块，使用 `get_db()`、`save_model_record()` 等函数。

### Q: 扩展可以修改平台核心代码吗？

A: 不推荐。扩展应通过钩子和 API 与平台交互，不要直接修改核心文件。

### Q: 扩展可以有自己的配置文件吗？

A: 可以。扩展可以在自己的目录下创建配置文件，通过 `manifest.json` 的 `frontend.templates` 添加配置页面。

### Q: 如何让扩展在管理面板中显示？

A: 在 `manifest.json` 中设置 `"admin_panel": true`，并添加 `frontend.templates` 配置页面。

### Q: 扩展支持多语言吗？

A: 支持。扩展可以在 `static/` 目录下放置多语言文件，通过前端 JS 动态切换。

---

> 本文档最后更新于 2026年7月