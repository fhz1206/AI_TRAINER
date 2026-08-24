<div align="center">

<img src="AI_TRAINER-logo.png" alt="AI_TRAINER Logo" width="260"/>

# 统一训练平台

</div>

> 基于 Flask + PyTorch 的深度学习训练与测试平台，支持 CNN 图像分类和 Transformer 文本生成模型的全流程训练、管理与测试。

> 用户和管理员的使用说明请查看使用说明查找表

> **生态联动**：AI_TRAINER <img src="AI_TRAINER-logo.png" width="18"/> × [CodeMate](https://gitcode.com/fhz1206/CodeMate) <img src="CodeMate-logo.png" width="18"/> × [CostCut-Infer](https://gitcode.com/fhz1206/CostCut-Infer) <img src="CostCut-Infer-logo.png">
---
# 使用说明查找表
| 用户使用说明 | 管理员使用说明 |
|------|------|
| [docs/DirectionsUser.md](docs/DirectionsUser.md) | [docs/DirectionsAdmin.md](docs/DirectionsAdmin.md) |
---

# 预计更新
## v1.2.0 添加扩展功能，当前将会兼容CodeMate项目，不过该项目也将会适配本项目（coding 1/2）

# 已经更新
## v1.0.1 修复一堆问题
修复切换页面后训练进度丢失的问题；
修复分词部分空格没有分token的问题；
修复admin页面无直接跳转至学习等页面的问题；
修复测试页面部分下拉框背景颜色问题；
修复admin页面部分内容宽度异常的问题；
修复admin页面实时服务器负载数据显示的线条不够柔顺的问题，优化了用户体验； 
修复admin页面搜索时`|`在emoji符号前的问题；
修复web页面的icon异常的问题，修复admin页面转到别的页面的路口不全的问题
优化页面背景
## v1.1.0 添加队列管理系统和资源限制系统
---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask |
| 深度学习 | PyTorch + torchvision |
| 图像处理 | Pillow, OpenCV |
| 手势识别 | MediaPipe |
| 系统监控 | psutil |
| 数据库 | SQLite3 (WAL 模式) |
| 分词 | HuggingFace Tokenizers (BPE) |
| 前端 | HTML5 + CSS3 + Vanilla JS |
| 进度条 | tqdm |
| 打包部署 | PyInstaller + Inno Setup |

---

## 项目结构

```
Windows_ITPP/
├── app.py                  # 应用入口（启动进度条 + 服务启动）
├── app_logger.py           # 全局请求日志（轮转 + 内存友好）
├── config.py               # 全局配置
├── database.py             # SQLite3 数据库管理
├── model.py                # 数据管线（Dataset/LRU缓存）+ 兼容门面
├── model_io.py             # 模型存取统一入口（safetensors + 元数据旁车）
├── trainer.py              # 兼容垫片（训练实现已迁移 trainers 包）
├── state.py                # 训练任务状态（内存）
├── cleanup.py              # 存储清理机制
├── requirements.txt        # Python 依赖
├── architectures/          # 🧱 模型积木库（可插拔）
│   ├── attention.py        #   注意力注册表：full / flash(默认) / linear
│   ├── blocks.py           #   TransformerBlock / 双向编码块(ViT)
│   └── moe.py              #   MoE 专家混合层（修复版）
├── models/                 # 模型实现包（按架构分文件存放）
│   ├── vision.py           #   CNN(SimpleResNet) / ViT 图像分类
│   ├── text.py             #   Decoder-only 文本生成（积木式注意力+MoE）
│   ├── diffusion.py        #   扩散生成 DDPM / 扩散编辑适配
│   ├── multimodal.py       #   多模态单流（图文 Decoder-only）
│   └── legacy.py           #   旧版组件存档（仅供历史 .pth 反序列化）
├── trainers/               # 训练器包（按训练类型分发）
│   ├── image_cls.py        #   图像分类（CNN/ViT 统一入口）
│   ├── text_gen.py         #   文本生成训练
│   ├── diffusion.py        #   扩散生成/编辑训练
│   └── multimodal.py       #   多模态图文配对训练
├── blueprints/
│   ├── auth.py             # 认证 API
│   ├── main.py             # 页面路由
│   ├── training.py         # 训练 API（含 /api/architecture_options 积木选项）
│   ├── model.py            # 模型管理 API
│   ├── test.py             # 测试 API
│   ├── profile.py          # 账号主页 API
│   ├── admin.py            # 管理员面板 API
│   └── utils.py            # 公共工具
├── static/
│   ├── css/style.css       # 全局样式
│   ├── css/ai.css          # 学习页样式
│   ├── js/main.js          # 训练页逻辑
│   ├── js/ai.js            # 学习页逻辑
│   └── tokenizer_bpe.json  # BPE 分词模型（精简版）
├── templates/
│   ├── homepage.html       # 首页（公开）
│   ├── index.html          # 训练面板（LLM/图像/多模态三分区）
│   ├── login.html          # 登录/注册
│   ├── check.html          # 模型测试
│   ├── ai.html             # AI 基础学习
│   ├── profile.html        # 账号主页
│   └── admin.html          # 管理员面板
├── docs/
│   ├── DirectionsUser.md   # 用户使用说明
│   ├── DirectionsAdmin.md  # 管理员使用说明
│   └── EXTENSION_DEV.md    # 扩展开发指南（CodeMate 联动）
├── uploads/                # 训练数据（运行时生成，不入库）
├── models/                 # 模型文件产物（运行时生成，不入库；代码包见上）
├── test_data/              # 测试数据（运行时生成，不入库）
└── training_platform.db    # SQLite 数据库（运行时生成，不入库）
```

### 模型架构总览

| 分区 | 架构 | 说明 |
|------|------|------|
| 大语言模型 | Transformer (Decoder-only) | 注意力积木可切换 full/flash(默认)/linear，可选 MoE、MLA |
| 图像模型 | CNN | 经典卷积分类 |
| 图像模型 | ViT | 视觉 Transformer 分类，注意力积木可切换 |
| 图像模型 | Diffusion (DDPM) | 图像生成，噪声预测式扩散 |
| 图像模型 | Diffusion Edit Adapter | 图像编辑，条件拼接 + 退化对自监督 |
| 多模态 | Single-Stream Decoder | 图文 token 单流拼接，看图续写 |

扩展新注意力或新模型：分别用 `@register_attention('名字')` / `@register_model('名字')`
注册即可自动出现在前端积木选项中，无需改动训练器与页面代码。

---

## 快速开始

### 环境要求

- Python 3.11+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python app.py
```

启动后访问 **http://127.0.0.1:5000** 即可使用。

---

## 页面路由

| 路由 | 页面 | 登录要求 |
|------|------|----------|
| `/home` | 首页（公开 Landing Page） | 否 |
| `/train` | 训练面板 | 是 |
| `/login` | 登录页 | 否 |
| `/register` | 注册页（默认注册标签） | 否 |
| `/check` | 模型测试页 | 是 |
| `/study` | AI 基础学习页 | 是 |
| `/profile` | 账号主页 | 是 |
| `/admin` | 管理员面板（仅 admin 账户） | 是 |

## API 接口

| 方法 | 路由 | 说明 |
|------|------|------|
| POST | `/api/register` | 注册 |
| POST | `/api/login` | 登录 |
| GET | `/api/user_info` | 用户信息 |
| POST | `/api/preview_zip` | 预览 ZIP |
| POST | `/api/upload` | 上传数据 |
| GET | `/api/list_datasets` | 数据集列表 |
| POST | `/api/start_training` | 启动训练 |
| GET | `/api/task_status/<id>` | 训练状态 |
| GET | `/api/list_models` | 模型列表 |
| GET | `/api/download_model/<name>` | 下载模型 |
| DELETE | `/api/delete_model/<name>` | 删除模型 |
| GET | `/api/list_user_models` | 测试模型列表 |
| POST | `/api/upload_test_data` | 上传测试数据 |
| POST | `/api/run_test` | 运行测试 |
| GET | `/api/profile/info` | 账号信息 |
| POST | `/api/profile/update_username` | 修改用户名 |
| POST | `/api/profile/update_password` | 修改密码 |
| GET | `/api/profile/activity_logs` | 行为日志 |
| GET | `/admin/api/users` | 用户列表（管理员） |
| POST | `/admin/api/user/<id>/role` | 修改用户角色 |
| POST | `/admin/api/user/<id>/group` | 修改用户分组 |
| POST | `/admin/api/user/<id>/delete` | 删除用户 |
| POST | `/admin/api/user/<id>/reset_password` | 重置用户密码 |
| GET | `/admin/api/logs` | 全部行为日志 |
| GET | `/admin/api/device_status` | 设备状态信息 |
| GET | `/admin/api/monitor` | 实时 CPU/RAM 监控 |
| GET | `/admin/api/queue` | 训练队列状态 |
| POST | `/admin/api/queue/max_concurrent` | 设置最大并发数 |
| POST | `/admin/api/queue/cancel/<id>` | 取消/停止训练任务 |
| GET | `/admin/api/bandwidth` | 带宽使用统计 |
| POST | `/admin/api/bandwidth/default` | 设置默认带宽限制 |
| POST | `/admin/api/bandwidth/user/<id>` | 设置用户带宽限制 |

---

## 数据库

| 表名 | 说明 |
|------|------|
| users | 用户信息（含 role 角色、group_name 分组） |
| files | 上传文件记录 |
| models | 训练模型记录 |
| activity_logs | 行为日志 |

数据库初始化时自动创建管理员账户 `admin / 123456`。

---

## 许可证

BSD 3-Clause