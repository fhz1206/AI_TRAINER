# 🧠 ITPP — 统一训练平台 (Integrated Training Platform)

> 一个基于 Flask + PyTorch 的深度学习训练与测试平台，支持 **CNN 图像分类** 和 **Transformer 生成式文本模型** 的全流程训练、管理与测试。

---

## 📑 目录

- [项目概览](#项目概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [后端 API 接口文档](#后端-api-接口文档)
  - [认证模块 (auth)](#认证模块-auth)
  - [训练模块 (training)](#训练模块-training)
  - [模型管理模块 (model)](#模型管理模块-model)
  - [测试模块 (test)](#测试模块-test)
  - [页面路由 (main)](#页面路由-main)
- [前端架构](#前端架构)
  - [页面布局](#页面布局)
  - [整体风格](#整体风格)
  - [JS 功能模块](#js-功能模块)
- [核心算法](#核心算法)
- [数据库设计](#数据库设计)
- [打包部署](#打包部署)

---

## 项目概览

ITPP 是一个面向深度学习初学者和研究者的 **一体化训练平台**，提供从数据上传、模型配置、训练执行到测试验证的完整工作流：

```
用户注册/登录 → 上传训练数据 → 配置模型参数 → 启动训练（异步） → 实时监控进度 → 下载模型 → 测试验证
```

**核心特性**：
- 🖼️ **CNN 图像分类**：基于 SimpleResNet 架构，自动推断类别数
- 📝 **Transformer 文本生成**：Decoder-only 架构，支持 MLA 低秩注意力
- 🔄 **异步训练**：训练任务在独立线程执行，前端轮询实时进度
- 🧪 **在线测试**：支持自定义测试代码，子进程安全执行
- 🛡️ **数值稳定性**：8 大防护机制（梯度裁剪、nan 检测、自动 Checkpoint 等）
- 📦 **桌面部署**：PyInstaller + Inno Setup 打包为 Windows 桌面应用

---

## 技术栈

| 层级 | 技术 | 版本/说明 |
|------|------|----------|
| **后端框架** | Flask | Python Web 框架，负责路由和 API |
| **跨域支持** | Flask-CORS | 支持跨域请求 |
| **深度学习** | PyTorch | 模型定义、训练、推理 |
| **计算机视觉** | torchvision | 图像预处理（Resize、Normalize） |
| **图像处理** | Pillow (PIL) | 图片加载与转换 |
| **数据库** | SQLite3 | 轻量级关系型数据库，WAL 模式 |
| **密码安全** | hashlib (SHA-256) | 密码哈希存储 |
| **前端** | HTML5 + CSS3 + Vanilla JS | 无框架，原生实现 |
| **字体** | Google Fonts (Inter) | 现代无衬线字体 |
| **进度条** | tqdm | 启动时模块加载进度展示 |
| **打包** | PyInstaller + Inno Setup | Windows 桌面应用打包 |

---

## 项目结构

```
Windows_ITPP/
├── app.py                  # 应用入口，带启动进度条
├── config.py               # 全局配置（目录、密钥、上传限制）
├── config.json             # PyInstaller 打包配置
├── database.py             # SQLite3 数据库管理（用户/文件/模型 CRUD）
├── model.py                # 神经网络模型定义 + 数据集类
├── trainer.py              # 训练器（CNN + Transformer，含数值稳定性保护）
├── state.py                # 全局训练任务状态（内存字典）
├── blueprints/
│   ├── __init__.py         # 蓝图包初始化
│   ├── auth.py             # 认证 API（注册/登录/用户信息）
│   ├── main.py             # 页面路由（首页/登录/退出/测试页）
│   ├── training.py         # 训练 API（上传/预览/数据集/启动训练/状态轮询）
│   ├── model.py            # 模型管理 API（列表/下载/删除）
│   ├── test.py             # 测试 API（模型列表/上传测试数据/运行测试）
│   └── utils.py            # 公共工具（登录装饰器/用户信息获取）
├── static/
│   ├── css/style.css       # 全局样式表（23.7KB）
│   └── js/main.js          # 训练页前端逻辑（26.4KB）
├── templates/
│   ├── index.html          # 训练主页面
│   ├── login.html          # 登录/注册页面
│   └── check.html          # 模型测试页面
├── uploads/                # 用户上传的训练数据
├── models/                 # 训练产出的模型文件（.pth）
├── test_data/              # 测试数据集
├── output/                 # 打包输出目录
└── training_platform.db    # SQLite 数据库文件
```

---

## 快速开始

### 环境要求

- Python 3.8+
- pip

### 安装依赖

```bash
pip install flask flask-cors torch torchvision pillow tqdm
```

### 启动服务

```bash
python app.py
```

启动后会显示进度条，完成后访问 **http://127.0.0.1:5000** 即可使用。

---

## 后端 API 接口文档

所有 API 前缀为 `/api`，需登录的接口使用 `@login_required` 装饰器保护（未登录重定向到 `/login`）。

---

### 认证模块 (auth)

**蓝图**：`auth_bp`，前缀 `/api`  
**源文件**：`blueprints/auth.py`  
**依赖库**：Flask (session, request, jsonify), database (register_user, verify_user, get_user_by_id, get_file_count, get_model_count)

#### `POST /api/register` — 用户注册

**功能**：注册新用户账号

**请求体** (JSON)：
```json
{
    "username": "testuser",
    "password": "1234"
}
```

**请求字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| username | string | ✅ | 用户名，2-20个字符 |
| password | string | ✅ | 密码，至少4个字符 |

**成功响应** (200)：
```json
{
    "status": "success",
    "message": "注册成功"
}
```

**失败响应** (400)：
```json
{
    "status": "error",
    "message": "用户名已存在"
}
```

**后端校验逻辑**：
- 用户名和密码不能为空
- 用户名长度 2-20 个字符
- 密码长度至少 4 个字符
- 用户名唯一性检查（SQLite IntegrityError）
- 密码使用 SHA-256 哈希存储

---

#### `POST /api/login` — 用户登录

**功能**：验证用户凭据，创建会话

**请求体** (JSON)：
```json
{
    "username": "testuser",
    "password": "1234"
}
```

**成功响应** (200)：
```json
{
    "status": "success",
    "message": "登录成功",
    "user": {
        "id": 1,
        "username": "testuser"
    }
}
```

**失败响应** (401)：
```json
{
    "status": "error",
    "message": "用户名或密码错误"
}
```

**后端逻辑**：
- 验证用户名密码后，将 `user_id` 和 `username` 写入 Flask session
- 设置 `session.permanent = True` 持久化会话
- 更新用户 `last_login` 时间戳

---

#### `GET /api/user_info` — 获取当前用户信息

**功能**：返回当前登录用户的详细信息及统计数据  
**需要登录**：✅

**成功响应** (200)：
```json
{
    "status": "success",
    "user": {
        "username": "testuser",
        "created_at": "2024-01-01 12:00:00",
        "last_login": "2024-01-15 09:30:00",
        "file_count": 3,
        "model_count": 2
    }
}
```

---

### 训练模块 (training)

**蓝图**：`training_bp`，前缀 `/api`  
**源文件**：`blueprints/training.py`  
**依赖库**：Flask, werkzeug (secure_filename), zipfile, tempfile, base64, config, database, state, trainer

#### `POST /api/preview_zip` — 预览 ZIP 文件结构

**功能**：解析上传的 ZIP 文件，返回类别列表和图片预览（base64 编码）  
**需要登录**：✅

**请求体** (multipart/form-data)：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| zip_file | File | ✅ | ZIP 压缩包文件 |

**成功响应** (200)：
```json
{
    "status": "success",
    "classes": [
        {
            "name": "猫",
            "count": 150,
            "samples": ["data:image/jpeg;base64,/9j/4AAQ...", "data:image/jpeg;base64,..."]
        },
        {
            "name": "狗",
            "count": 200,
            "samples": ["data:image/jpeg;base64,..."]
        }
    ],
    "total_images": 350,
    "total_classes": 2
}
```

**后端逻辑**：
1. 将 ZIP 保存到临时目录并解压
2. 遍历顶层文件夹，每个文件夹视为一个类别
3. 递归查找所有图片文件（.jpg/.jpeg/.png/.bmp/.gif）
4. 每个类别取前 4 张图片转为 base64 预览
5. 返回类别名称、图片数量和预览数据

---

#### `POST /api/upload` — 上传训练数据

**功能**：上传训练数据文件（图片 ZIP 或文本文件），保存到用户目录并记录到数据库  
**需要登录**：✅

**请求体** (multipart/form-data)：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | ✅ | 训练数据文件（ZIP/TXT） |
| train_type | string | ✅ | 训练类型：`"image"` 或 `"text"` |

**成功响应** (200)：
```json
{
    "status": "success",
    "message": "上传成功！共 150 个文件",
    "file_count": 150,
    "file_id": 5
}
```

**后端逻辑**：
1. 文件保存到 `uploads/{user_id}/{train_type}/` 目录
2. 如果是 ZIP 文件，自动解压到同名子目录
3. 递归统计有效文件数量（图片或文本）
4. 调用 `save_file_record()` 写入数据库
5. 返回文件记录 ID 供后续数据集选择使用

---

#### `GET /api/preview_data?train_type=image` — 预览已上传数据

**功能**：返回用户已上传数据的预览信息（图片 base64 / 文本内容）  
**需要登录**：✅

**查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| train_type | string | ✅ | `"image"` 或 `"text"` |

**成功响应** (200)：
```json
{
    "items": [
        {
            "type": "image",
            "name": "cat_001.jpg",
            "data": "data:image/jpeg;base64,/9j/4AAQ..."
        }
    ]
}
```

文本类型响应：
```json
{
    "items": [
        {
            "type": "text",
            "name": "article_001.txt",
            "preview": "这是一篇关于人工智能的文章..."
        }
    ]
}
```

**限制**：最多返回 12 个预览项

---

#### `GET /api/list_datasets` — 获取用户数据集列表

**功能**：返回当前用户所有已上传的训练数据集  
**需要登录**：✅

**成功响应** (200)：
```json
{
    "datasets": [
        {
            "id": 1,
            "name": "image_data",
            "path": "uploads/1/image/image_data",
            "sample_count": 350,
            "file_size": "125.3 MB",
            "created_at": "2024-01-15 10:30:00",
            "type": "image"
        }
    ]
}
```

**后端逻辑**：
- 从数据库查询用户所有文件记录
- 过滤掉文件路径不存在的记录
- 递归统计每个数据集的有效样本数
- 格式化文件大小显示

---

#### `POST /api/start_training` — 启动训练任务

**功能**：根据用户配置启动异步训练任务  
**需要登录**：✅

**请求体** (JSON)：
```json
{
    "train_type": "image",
    "dataset_id": 1,
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 32,
    "image_size": 224,
    "num_classes": 10,
    "base_channels": 64,
    "vocab_size": 1000,
    "max_seq_len": 128,
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "use_moe": false,
    "use_mla": false
}
```

**请求字段说明**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| train_type | string | "image" | 训练类型：`"image"` 或 `"text"` |
| dataset_id | int | — | 数据集 ID（优先使用） |
| learning_rate | float | 0.0001 | 学习率 |
| epochs | int | 10 | 训练轮数 |
| batch_size | int | 32 | 批次大小 |
| **CNN 专属参数** | | | |
| image_size | int | 224 | 输入图片尺寸 |
| num_classes | int | 10 | 分类数量（实际由数据集自动推断） |
| base_channels | int | 64 | 基础通道数 |
| **Transformer 专属参数** | | | |
| vocab_size | int | 1000 | 词汇表大小 |
| max_seq_len | int | 128 | 最大序列长度 |
| d_model | int | 512 | 模型隐层维度 |
| n_layers | int | 6 | Transformer 层数 |
| n_heads | int | 8 | 注意力头数 |
| d_ff | int | 2048 | 前馈网络维度 |
| dropout | float | 0.1 | Dropout 比率 |
| use_moe | bool | false | 是否启用 MoE（当前已禁用） |
| use_mla | bool | false | 是否启用 MLA 低秩注意力 |

**成功响应** (200)：
```json
{
    "status": "success",
    "task_id": "0473d101",
    "message": "image 训练已启动 (task_id: 0473d101)"
}
```

**后端逻辑**：
1. 生成 8 位随机 task_id
2. 根据 dataset_id 查找数据集路径，若无则使用用户最新数据集
3. 构建模型参数和训练参数字典
4. 创建守护线程执行 `train_image_model()` 或 `train_text_model()`
5. 立即返回 task_id，前端通过轮询获取进度

---

#### `GET /api/task_status/<task_id>` — 查询训练任务状态

**功能**：实时查询指定训练任务的进度和指标  
**需要登录**：✅

**路径参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | string | 训练任务 ID |

**训练中响应** (200)：
```json
{
    "status": "running",
    "progress": 65,
    "loss": 0.3245,
    "accuracy": 89.5,
    "message": "Epoch 3/10 | Batch 15/30 | Loss: 0.3245 | Acc: 89.50%"
}
```

**完成响应**：
```json
{
    "status": "completed",
    "progress": 100,
    "loss": 0.1234,
    "accuracy": 95.2,
    "message": "✅ 训练完成！准确率: 95.20% | 损失: 0.1234 | 耗时: 120s | 模型文件: cnn_0473d101_1705312000.pth"
}
```

**失败响应**：
```json
{
    "status": "failed",
    "progress": 0,
    "loss": null,
    "accuracy": null,
    "message": "❌ 训练失败: 梯度爆炸..."
}
```

**状态值说明**：

| status | 说明 |
|--------|------|
| running | 训练进行中 |
| completed | 训练完成 |
| failed | 训练失败 |
| not_found | 任务不存在（返回 404） |

---

### 模型管理模块 (model)

**蓝图**：`model_bp`，前缀 `/api`  
**源文件**：`blueprints/model.py`  
**依赖库**：Flask (send_file), os.path, urllib.parse, database

#### `GET /api/list_models` — 列出用户模型

**功能**：返回当前用户所有训练完成的模型列表  
**需要登录**：✅

**成功响应** (200)：
```json
{
    "models": [
        {
            "name": "cnn_0473d101_1705312000.pth",
            "size": "98.50 MB",
            "type": "🖼️ CNN",
            "accuracy": 95.2,
            "loss": 0.1234,
            "created_at": "2024-01-15 12:00:00"
        },
        {
            "name": "text_gen_36a695be_1705312100.pth",
            "size": "98.50 MB",
            "type": "📝 Transformer",
            "accuracy": 15.3,
            "loss": 2.5678,
            "created_at": "2024-01-15 14:30:00"
        }
    ]
}
```

**后端逻辑**：
- 优先从数据库查询模型记录
- 若数据库无记录，兼容从磁盘 `models/{user_id}/` 目录扫描 .pth 文件
- 仅返回文件仍然存在的模型

---

#### `GET /api/download_model/<filename>` — 下载模型

**功能**：下载指定的模型文件  
**需要登录**：✅

**路径参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| filename | string | 模型文件名 |

**响应**：文件流下载（`Content-Disposition: attachment`）

**错误响应** (404)：
```json
{
    "status": "error",
    "message": "文件不存在"
}
```

---

#### `DELETE /api/delete_model/<filename>` — 删除模型

**功能**：删除指定模型文件及数据库记录  
**需要登录**：✅

**路径参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| filename | string | 模型文件名 |

**成功响应** (200)：
```json
{
    "status": "success",
    "message": "模型已删除"
}
```

**后端逻辑**：
1. 删除磁盘文件 `models/{user_id}/{filename}`
2. 删除数据库中的模型记录

---

### 测试模块 (test)

**蓝图**：`test_bp`，前缀无（路由直接 `/api/...`）  
**源文件**：`blueprints/test.py`  
**依赖库**：Flask, shutil, subprocess, tempfile, re, zipfile, ast, werkzeug (secure_filename), config

#### `GET /api/list_user_models?framework=image` — 获取用户模型（按框架过滤）

**功能**：返回当前用户的模型列表，可按框架类型过滤  
**需要登录**：✅

**查询参数**：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| framework | string | ❌ | "all" | `"image"` / `"text"` / `"all"` |

**成功响应** (200)：
```json
{
    "status": "success",
    "models": [
        {
            "name": "cnn_0473d101_1705312000.pth",
            "type": "cnn",
            "path": "models/1/cnn_0473d101_1705312000.pth",
            "size": "98.5 MB"
        }
    ]
}
```

**框架映射逻辑**：
- `framework=image` → 过滤模型名包含 `cnn` 的模型
- `framework=text` → 过滤模型名包含 `text_gen` 或 `transformer` 的模型
- `framework=all` → 返回所有模型

---

#### `POST /api/upload_test_data` — 上传测试数据集

**功能**：上传测试用的数据集文件  
**需要登录**：✅

**请求体** (multipart/form-data)：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | ✅ | 测试数据文件 |
| framework | string | ✅ | 框架类型：`"image"` 或 `"text"` |

**文件格式要求**：
- 图片测试：仅支持 `.zip` 格式
- 文本测试：支持 `.txt` 或 `.zip` 格式

**成功响应** (200)：
```json
{
    "status": "success",
    "message": "上传成功",
    "file_count": 50,
    "path": "test_data/1/test_1705312000_test_data",
    "filename": "test_data.zip"
}
```

**后端逻辑**：
1. 保存到 `test_data/{user_id}/` 目录
2. ZIP 文件自动解压（含有效性校验和冲突处理）
3. 递归统计文件数量

---

#### `POST /api/run_test` — 运行测试

**功能**：执行用户自定义的测试代码，返回运行结果  
**需要登录**：✅

**请求体** (JSON)：
```json
{
    "framework": "image",
    "model_name": "cnn_0473d101_1705312000.pth",
    "test_code": "import torch\nmodel = torch.load(model_path, ...)\nprint('测试结果')",
    "test_data_path": "test_data/1/test_1705312000_data"
}
```

**请求字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| framework | string | ✅ | `"image"` 或 `"text"` |
| model_name | string | ✅ | 模型文件名 |
| test_code | string | ✅ | 用户编写的 Python 测试代码 |
| test_data_path | string | 图片必填 | 测试数据路径（文本框架可为空） |

**成功响应** (200)：
```json
{
    "status": "success",
    "output": "✅ 测试完成\n准确率：92.50%",
    "error": "",
    "metrics": {
        "accuracy": 92.5
    }
}
```

**失败响应** (200)：
```json
{
    "status": "failed",
    "output": "",
    "error": "NameError: name 'x' is not defined",
    "metrics": {}
}
```

**后端执行流程**：
1. **参数校验**：图片框架必须有 test_data_path，模型文件必须存在
2. **代码注入**：在用户代码前注入公共变量：
   ```python
   import sys
   sys.path.insert(0, '.')
   sys.stdout.reconfigure(encoding='utf-8')
   sys.stderr.reconfigure(encoding='utf-8')
   model_path = 'models/1/cnn_xxx.pth'
   test_data_path = 'test_data/1/...'
   framework = 'image'
   ```
3. **语法校验**：使用 `ast.parse()` 检查语法错误，精确定位到用户代码行号
4. **子进程执行**：写入临时 `.py` 文件，通过 `subprocess.run()` 执行（超时 300 秒）
5. **指标解析**：正则匹配输出中的准确率、Loss、困惑度（PPL）
6. **清理**：删除临时脚本文件

**可解析的指标格式**：
- 准确率：`准确率：92.50%` 或 `准确率: 92.50%`
- Loss：`Loss：0.1234` 或 `Loss: 0.1234`
- 困惑度：`困惑度（PPL）：15.3` 或 `困惑度（PPL）: inf`

---

### 页面路由 (main)

**蓝图**：`main_bp`，无前缀  
**源文件**：`blueprints/main.py`

| 路由 | 方法 | 需登录 | 功能 |
|------|------|--------|------|
| `GET /` | GET | ✅ | 训练主页面，渲染 `index.html` |
| `GET /login` | GET | ❌ | 登录页面，渲染 `login.html`（已登录则重定向到 `/`） |
| `GET/POST /logout` | GET/POST | ❌ | 退出登录（清除 session） |
| `GET /check` | GET | ✅ | 模型测试页面，渲染 `check.html` |

---

## 前端架构

### 页面布局

#### 1. 登录/注册页面 (`login.html`)

```
┌─────────────────────────────────────┐
│            🧠 训练平台               │
│     登录以管理您的训练任务与模型       │
│                                     │
│    ┌──────────┬──────────┐          │
│    │  登录    │  注册    │  ← 标签切换│
│    └──────────┴──────────┘          │
│                                     │
│    用户名: [________________]        │
│    密  码: [________________]        │
│                                     │
│    [       🚀 登录       ]          │
│                                     │
│    还没有账号？立即注册               │
└─────────────────────────────────────┘
```

- 居中卡片布局，最大宽度 420px
- 登录/注册标签切换，共享同一消息提示区
- 渐变紫色提交按钮，悬停上浮 + 阴影效果
- 支持 Enter 键自动提交

#### 2. 训练主页面 (`index.html`)

```
┌──────────────────────────────────────────────────┐
│  🧠 统一训练平台          username  🧪测试  🚪退出 │ ← 顶部栏
├──────────────────────────────────────────────────┤
│  [🖼️ 图片训练]  [📝 文字训练]                     │ ← 类型切换标签
├──────────────────────────────────────────────────┤
│  📁 上传训练数据                     [未上传]      │
│  ┌──────────────────────────────────────┐        │
│  │  📋 数据集结构要求说明               │        │
│  │  📤 点击或拖拽上传区域               │        │
│  │  📂 ZIP 结构预览（上传后显示）       │        │
│  │  [✅ 确认上传数据集]                 │        │
│  └──────────────────────────────────────┘        │
├──────────────────────────────────────────────────┤
│  🖼️ CNN 图片模型参数              [活跃]         │
│  图片尺寸[224]  分类数量[10]  基础通道[64]        │ ← 3列参数网格
├──────────────────────────────────────────────────┤
│  🗂️ 选择训练数据集                 [3 个数据集]   │
│  ┌─ ✅ image_data (350样本, 图片) ──────┐        │
│  │  📝 text_data  (50样本, 文本)        │        │
│  └──────────────────────────────────────┘        │
├──────────────────────────────────────────────────┤
│  ⚡ 训练超参数                                    │
│  学习率[0.0001]  训练轮数[10]  批次大小[32]       │
├──────────────────────────────────────────────────┤
│  [            🚀 开始训练            ]            │ ← 主操作按钮
├──────────────────────────────────────────────────┤
│  📊 训练状态                        [🔄 运行中]   │
│  ████████████████░░░░░░░░░  65%                   │ ← 进度条
│  Loss: 0.3245  |  准确率: 89.50%  |  进度: 65%   │ ← 指标卡片
├──────────────────────────────────────────────────┤
│  💾 已保存的模型                        [2]       │
│  🖼️ cnn_xxx.pth (98.5MB)  [⬇️ 下载]            │
│  📝 text_gen_xxx.pth (98.5MB) [⬇️ 下载]         │
└──────────────────────────────────────────────────┘
```

- 单列垂直布局，最大宽度 900px
- 卡片式分区，每个功能区独立卡片
- 参数使用 3 列网格布局
- 训练状态区包含进度条 + 3 个指标卡片

#### 3. 模型测试页面 (`check.html`)

```
┌──────────────────────────────────────────────────────────┐
│  🧪 模型测试平台            username  🏠训练平台  🚪退出  │
├──────────────────────────────────────────────────────────┤
│  模型框架: [CNN 图片模型 ▾]  选择模型: [cnn_xxx ▾] [🔄] │
├───────────────────────────────┬──────────────────────────┤
│  💻 测试代码                  │  📊 测试结果              │
│  ┌──┬────────────────────┐   │  ┌──────────────────────┐│
│  │ 1│ import torch       │   │  │ ✅ 测试成功           ││
│  │ 2│ model = torch.load │   │  │ ---------- 运行日志 - ││
│  │ 3│ model.eval()       │   │  │ 正在加载模型...       ││
│  │ 4│ # 测试代码         │   │  │ ✅ 准确率: 92.50%     ││
│  │ 5│ ...                │   │  │ ---------- 测试指标 - ││
│  │  │                    │   │  │ accuracy: 92.5       ││
│  └──┴────────────────────┘   │  └──────────────────────┘│
│  [▶️ 运行测试] [🗑️ 清空结果] │                          │
│                               │                          │
│  📁 上传测试数据集 (CNN时显示) │                          │
│  📤 点击或拖拽上传             │                          │
└───────────────────────────────┴──────────────────────────┘
```

- 双列布局：左侧代码+上传（3fr），右侧结果（2fr）
- 代码编辑器带行号 + Python 语法高亮（Dracula 配色）
- CNN 模式显示上传区，Transformer 模式自动隐藏
- 结果区支持成功/失败/信息三种颜色标注

---

### 整体风格

**设计语言**：深色科技风（Dark Tech）

| 元素 | 规范 |
|------|------|
| **主背景** | 深蓝渐变 `#0f0f1a → #1a1a2e → #16213e` |
| **卡片背景** | 半透明毛玻璃 `rgba(255,255,255,0.05)` + `backdrop-filter: blur(20px)` |
| **主色调** | 靛蓝紫 `#6366f1`（按钮、进度条、焦点边框） |
| **文字颜色** | 主文字 `#ffffff`，次要 `rgba(255,255,255,0.6)`，弱化 `rgba(255,255,255,0.3)` |
| **边框** | `rgba(255,255,255,0.1)`，悬停加深 |
| **圆角** | 小 `8px`，中 `12px`，大 `24px` |
| **阴影** | `0 25px 50px rgba(0,0,0,0.5)` |
| **字体** | Inter（Google Fonts），代码区 Consolas/Monaco |
| **交互** | 按钮悬停上浮 `translateY(-2px)` + 阴影扩散，0.3s 过渡动画 |
| **图标** | Emoji 图标（🧠🖼️📝📁📊💾🚪等），无额外图标库 |

**代码编辑器配色**（Dracula 风格）：

| 语法元素 | 颜色 | 色值 |
|---------|------|------|
| 关键字 (import, def, if) | 亮粉 | `#ff79c6` |
| 内置函数 (print, len) | 亮绿 | `#50fa7b` |
| 字符串 | 亮橙 | `#ffb86c` |
| 注释 | 亮青 | `#8be9fd` |
| 数字 | 亮紫 | `#bd93f9` |
| 函数调用 | 亮黄 | `#f1fa8c` |
| self | 亮青 | `#8be9fd` |
| 运算符 | 亮粉 | `#ff79c6` |
| 默认文本 | 亮白 | `#e0e0e0` |
| 行号栏背景 | 深蓝 | `#1a1a2e` |
| 行号文字 | 灰色 | `#858585` |

---

### JS 功能模块

#### `main.js` — 训练主页面逻辑 (695行)

**全局状态管理**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `currentType` | string | 当前训练类型：`'image'` 或 `'text'` |
| `currentTaskId` | string | 当前训练任务 ID |
| `statusInterval` | number | 轮询定时器 ID |
| `uploadedFileCount` | number | 已上传文件数量 |
| `selectedDatasetId` | number | 当前选中的数据集 ID |
| `selectedClassFile` | File | 选中的图片 ZIP 文件 |
| `classPreviewData` | object | ZIP 预览数据 |
| `selectedTextFile` | File | 选中的文本文件 |

**功能函数**：

| 函数 | 功能 | 调用的 API |
|------|------|-----------|
| `switchType(type)` | 切换图片/文本训练类型，显示对应参数面板和上传区 | — |
| `handleClassFileSelect(file)` | 处理图片 ZIP 选择，调用预览接口 | `POST /api/preview_zip` |
| `renderClassPreview(data)` | 渲染 ZIP 类别预览卡片和缩略图 | — |
| `confirmClassUpload()` | 确认上传图片数据集 | `POST /api/upload` |
| `handleTextFileSelect(file)` | 处理文本文件选择和上传 | `POST /api/upload` |
| `loadDatasets()` | 加载用户数据集列表并渲染 | `GET /api/list_datasets` |
| `selectDataset(id)` | 切换选中数据集 | — |
| `startTraining()` | 收集参数并启动训练 | `POST /api/start_training` |
| `pollStatus()` | 每 500ms 轮询训练状态 | `GET /api/task_status/{id}` |
| `loadModels()` | 加载已保存模型列表 | `GET /api/list_models` |
| `downloadModel(name)` | 下载模型文件 | `GET /api/download_model/{name}` |
| `handleLogout()` | 退出登录 | `POST /logout` |

**训练流程时序**：

```
用户点击"开始训练"
    ↓
startTraining() → 校验数据集 → 收集参数 → POST /api/start_training
    ↓
获取 task_id → 启动轮询 (setInterval 500ms)
    ↓
pollStatus() → GET /api/task_status/{id} → 更新进度条/指标/状态徽章
    ↓
status === 'completed' → 停止轮询 → 刷新模型列表
status === 'failed'    → 停止轮询 → 显示错误信息
```

#### `check.html` 内嵌 JS — 测试页面逻辑

**全局状态**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `currentFramework` | string | 当前框架：`'image'` 或 `'text'` |
| `selectedTestDataPath` | string | 测试数据路径 |
| `isRunning` | boolean | 是否正在运行测试 |

**功能函数**：

| 函数 | 功能 | 调用的 API |
|------|------|-----------|
| `onFrameworkChange()` | 切换框架，更新代码模板和上传区 | — |
| `updateUploadTip()` | 更新上传提示，控制上传卡片显隐 | — |
| `loadModels()` | 按框架加载模型列表 | `GET /api/list_user_models?framework=` |
| `bindUploadEvents()` | 绑定测试数据上传的点击和拖拽事件 | — |
| `handleTestFileSelect(file)` | 处理测试文件上传 | `POST /api/upload_test_data` |
| `runTest()` | 执行测试 | `POST /api/run_test` |
| `resetCode()` | 恢复默认代码模板 | — |
| `bindCodeEditor()` | 绑定编辑器事件（Tab/输入/滚动） | — |

**语法高亮引擎**：

| 函数 | 功能 |
|------|------|
| `escH(t)` | HTML 转义（&, <, >） |
| `hlLine(line)` | 单行 Python 语法高亮解析 |
| `hlPy(code)` | 多行代码高亮（按行分割后逐行调用 hlLine） |
| `updateLineNumbers()` | 根据代码行数更新行号栏 |
| `updateHighlight()` | 刷新语法高亮层 |
| `syncScroll()` | 同步行号栏和高亮层的滚动位置 |
| `refreshEditor()` | 同时刷新行号和高亮 |

**编辑器实现原理**：
- 采用 **textarea + highlight-layer 叠加** 方案
- textarea 设置 `-webkit-text-fill-color: transparent` 使文字透明
- 底层 highlight-layer 显示着色后的 HTML
- 两者字体、行高、padding 完全一致，实现视觉对齐
- 用户在 textarea 中输入，高亮层实时同步渲染

---

## 核心算法

### SimpleResNet（CNN 图像分类）

```
输入 (3, 224, 224)
  ↓
Conv2d(3→64, 7×7, stride=2) + BN + ReLU + MaxPool(3×3, stride=2)
  ↓
Layer1: 2× ConvBlock(64→64)
  ↓
Layer2: 2× ConvBlock(64→128, stride=2)
  ↓
Layer3: 2× ConvBlock(128→256, stride=2)
  ↓
AdaptiveAvgPool2d(1, 1) + Flatten + Dropout + Linear(256→num_classes)
```

- Kaiming 正态初始化
- ImageNet 标准归一化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- LRU 缓存加速图像加载

### TextTransformerModel（生成式文本模型）

```
输入 token_ids (batch, seq_len)
  ↓
TokenEmbedding(vocab_size, d_model) + PositionEmbedding(max_seq_len, d_model)
  ↓
N × TransformerDecoderLayer(d_model, n_heads, d_ff, pre-norm, GELU)
  ↓  (因果掩码 + padding 掩码)
[MLA投影] (可选: d_model → mla_dim)
  ↓
LM_Head(d_model → vocab_size)  [权重共享 with TokenEmbedding]
  ↓
输出 logits (batch, seq_len, vocab_size)
```

- Pre-norm（先 LayerNorm 再注意力，数值更稳定）
- 权重共享（LM Head 与 Token Embedding 共享权重）
- GELU 激活函数
- MLA 低秩注意力（可选，投影到低维再映射回词表）
- MoE 已禁用（保留参数兼容前端）

### 数值稳定性保护（8大机制）

| # | 机制 | 位置 | 说明 |
|---|------|------|------|
| 1 | aux_loss 权重限制 | trainer.py | MoE 辅助损失权重上限 0.05 |
| 2 | 学习率上限 | trainer.py | 限制 3e-5，大学习率是梯度爆炸首因 |
| 3 | 输入参数检测 | trainer.py | 检测异常 token/标签，避免训练崩溃 |
| 4 | 每轮验证 | trainer.py | 每个 epoch 开始前验证模型输出是否有 nan/inf |
| 5 | logits 数值清洗 | trainer.py | `nan_to_num` 清洗异常 logits |
| 6 | 梯度爆炸检测+裁剪 | trainer.py | 检测 nan/inf 梯度并终止，`clip_grad_norm_` 裁剪 |
| 7 | 参数定期检测 | trainer.py | 每 100 步检测模型参数是否有 nan/inf |
| 8 | 自动 Checkpoint | trainer.py | 每 500 步保存 Checkpoint，避免丢进度 |

---

## 数据库设计

**数据库**：SQLite3 (`training_platform.db`)  
**模式**：WAL（Write-Ahead Logging），提高并发性能

### ER 关系图

```
users (1) ──────< (N) files
users (1) ──────< (N) models
```

### users 表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | INTEGER | PK, AUTO | 用户 ID |
| username | TEXT | UNIQUE, NOT NULL | 用户名 |
| password_hash | TEXT | NOT NULL | SHA-256 哈希密码 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 注册时间 |
| last_login | TIMESTAMP | NULL | 最后登录时间 |

### files 表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | INTEGER | PK, AUTO | 文件记录 ID |
| user_id | INTEGER | FK → users(id), CASCADE | 所属用户 |
| filename | TEXT | NOT NULL | 存储文件名 |
| original_name | TEXT | NOT NULL | 原始文件名 |
| train_type | TEXT | DEFAULT 'image' | 训练类型：image/text |
| file_size | INTEGER | DEFAULT 0 | 文件大小（字节） |
| file_path | TEXT | — | 文件存储路径 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 上传时间 |

### models 表

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | INTEGER | PK, AUTO | 模型记录 ID |
| user_id | INTEGER | FK → users(id), CASCADE | 所属用户 |
| model_name | TEXT | NOT NULL | 模型文件名 |
| model_type | TEXT | NOT NULL | 模型类型：cnn/text |
| file_size | INTEGER | DEFAULT 0 | 文件大小（字节） |
| file_path | TEXT | — | 模型文件路径 |
| accuracy | REAL | NULL | 准确率/困惑度 |
| loss | REAL | NULL | 最终损失值 |
| epochs | INTEGER | NULL | 训练轮数 |
| created_at | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 创建时间 |

---

## 打包部署

### PyInstaller 打包

项目使用 `auto-py-to-exe` 生成配置文件 `config.json`，打包为 Windows 桌面应用：

```bash
pyinstaller --noconfirm --onedir --console --icon icon.ico app.py
```

**打包包含的资源**：
- Python 源码：config.py, database.py, model.py, state.py, trainer.py
- 前端资源：templates/, static/, blueprints/
- 数据目录：uploads/, test_data/, models/
- 依赖库：flask, flask_cors, torch, PIL, tqdm, sqlite3 等

### Inno Setup 安装程序

`output/Inno.iss` 配置生成 Windows 安装程序，包含应用图标和安装向导。

---

## 许可证

MIT License

---

## 常见问题

### Q: 训练时出现"梯度爆炸/nan/inf"怎么办？
A: 平台已内置 8 大数值稳定性保护机制。如果仍然出现，建议：
1. 降低学习率（文本训练建议 ≤ 3e-5）
2. 减小 batch_size
3. 增大 dropout（但不超过 0.1）
4. 检查数据集是否有异常内容

### Q: Transformer 模型测试时生成结果为空怎么办？
A: 确保训练数据目录下存在词表文件 `*_token2char.pth`。测试模板会自动从三个位置查找词表：
1. 模型 `_metadata.train_params.data_path` 推断路径
2. 模型同目录的 `checkpoints_*` 子目录
3. `uploads` 目录全局搜索

### Q: 如何添加新的训练类型？
A: 需要修改以下文件：
1. `model.py` — 添加新模型类和数据集类
2. `trainer.py` — 添加新训练函数
3. `blueprints/training.py` — 添加新的启动逻辑
4. `templates/index.html` — 添加参数面板
5. `static/js/main.js` — 添加前端交互

### Q: 数据库文件在哪里？
A: `training_platform.db` 位于项目根目录，SQLite3 格式，可用任意 SQLite 工具查看。

### Q: 支持多用户同时训练吗？
A: 支持。每个用户的训练任务在独立线程中执行，数据按 `user_id` 隔离。但受限于 GIL，同一时刻只有一个线程在执行 PyTorch 计算。

本项目仅供学习和研究使用。
