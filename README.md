# 统一训练平台

> 基于 Flask + PyTorch 的深度学习训练与测试平台，支持 CNN 图像分类和 Transformer 文本生成模型的全流程训练、管理与测试。

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
├── app.py                  # 应用入口（启动进度条 + 全局请求日志）
├── config.py               # 全局配置
├── database.py             # SQLite3 数据库管理
├── model.py                # 模型定义 + 数据集类
├── trainer.py              # 训练器（CNN + Transformer）
├── state.py                # 训练任务状态（内存）
├── requirements.txt        # Python 依赖
├── blueprints/
│   ├── auth.py             # 认证 API
│   ├── main.py             # 页面路由
│   ├── training.py         # 训练 API
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
│   ├── index.html          # 训练面板
│   ├── login.html          # 登录/注册
│   ├── check.html          # 模型测试
│   ├── ai.html             # AI 基础学习
│   ├── profile.html        # 账号主页
│   └── admin.html          # 管理员面板
├── tokenizer/
│   └── tokenizer.json  # BPE 分词模型
├── uploads/                # 训练数据
├── models/                 # 模型文件 (.pth)
├── test_data/              # 测试数据
└── training_platform.db    # 数据库
```

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

MIT License