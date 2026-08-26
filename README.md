<div align="center">

<img src="AI_TRAINER-logo.png" alt="AI_TRAINER Logo" width="260"/>

# 统一训练平台

</div>

> 基于 Flask + PyTorch 的深度学习训练与测试平台：覆盖 **CNN / ViT 图像分类、扩散图像生成与编辑、Decoder-only 文本生成（LLM）、多模态图文单流** 六类模型的全流程训练、管理与测试。注意力机制（full / flash / linear）像搭积木一样在 Web 端自由切换，训练产物统一导出为 **Safetensors 标准包**（权重 + config.json + 分词器文件 + LICENSE）。

> [!WARNING]
> **⚠️ 自 v1.2.1 版本开始，将不再支持旧的 `.pth` 模型文件！**
> 平台已全面切换为 **Safetensors** 单一格式：模型权重统一为 `model.safetensors` +
> `config.json` 旁车元数据，词表统一为 `*_token2char.json`；模型列表、下载导出与
> 测试加载均只识别 `.safetensors`。如仍有历史 `.pth` 模型文件，请先在 v1.2.1 之前的
> 版本中迁移或重新训练后再升级。

> 用户和管理员的使用说明请查看使用说明查找表

> **生态联动**：[CodeMate](https://gitcode.com/fhz1206/CodeMate) <img src="CodeMate-logo.png" width="18"/> × [CostCut-Infer](https://gitcode.com/fhz1206/CostCut-Infer) <img src="CostCut-Infer-logo.png" width="72"/>
---
# 使用说明查找表
| 用户使用说明 | 管理员使用说明 |
|------|------|
| [docs/DirectionsUser.md](docs/DirectionsUser.md) | [docs/DirectionsAdmin.md](docs/DirectionsAdmin.md) |
---

# 更新日志

## v1.2.x 当前版本亮点
- **新增语言分类模型**：字符级 Transformer 双向编码器，按类别文件夹组织数据即可训练；
  LLM 上传数据放宽为同级散放 txt 或按文件夹组织均可
- **混合注意力积木搭建器**：LLM 训练页 Scratch 式拖拽"full/flash/linear"积木，
  逐层装配注意力；层数不足自动循环已搭序列、超出自动截断并提醒，支持首尾特殊层设置；
  **ViT 同样支持**统一注意力与拖拽式逐层混合注意力
- **扩散模型 DDIM 快速采样**：生成/编辑支持指定步数的 DDIM 确定性采样（训练页可选轻量子类型）
- **模型架构积木化**：新增 ViT 图像分类、扩散图像生成/编辑、多模态图文单流；
  注意力机制 full / flash(默认) / linear 在 Web 端像搭积木一样切换；MoE 修复广播 bug
- **AI 学习页**：新增 16+ 可视化章节（梯度下降二次函数演示、激活函数曲线、
  卷积扫描动画、温度采样等），侧边栏导航布局
- **标准模型导出包**：所有模型一键下载 Safetensors 标准结构
  （model.safetensors + config.json + vocab.json / merges.txt + 多模态视觉编码器权重 + LICENSE）
- **测试页全覆盖**：七类模型（含手部检测）均可选模型 → 自动填充本地模板 → 一键运行测试
- **资源占用上限**：默认取系统一半资源（CPU 线程 / 内存），超限拒绝新训练，管理端可调
- **存储治理**：轮转日志（单文件 5MB×5）、管理员自动/手动清理过期文件与日志；
  行为日志存储上限可调（默认 1000 条、-1 无上限，新增一条自动删除最旧一条），
  日志查询分页返回 + 索引优化，防止大结果撑爆内存
- **性能优化**：移除训练人工延时、DataLoader 多进程化（受限环境自动回退单进程）
- **CI/CD**：`.gitcode/workflows` 流水线自动执行全量编译检查与六类训练 E2E 回归

## 历史
- v1.1.0 队列管理系统与带宽限制系统
- v1.0.1 大量体验修复（进度保持、分词、admin 页面等）
---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask + flask-cors |
| 深度学习 | PyTorch + torchvision（可选 torch.compile 加速） |
| 模型格式 | SafeTensors（权重）+ config.json 元数据旁车 |
| 图像处理 | OpenCV（中文路径安全解码）、torchvision |
| 手势识别 | MediaPipe Tasks |
| 系统监控 | psutil（CPU/RAM 监控与资源上限） |
| 数据库 | SQLite3 (WAL 模式) |
| 分词 | 字符级词表(JSON) + 平台 BPE 分词器 |
| 前端 | HTML5 + CSS3 + Vanilla JS |
| CI/CD | GitCode Pipeline（`.gitcode/workflows`，E2E 回归自动化） |
| 打包部署 | PyInstaller + Inno Setup |

---

## 项目结构

```
Windows_ITPP/
├── app.py                  # 应用入口（启动进度条 + 服务启动）
├── app_logger.py           # 全局请求日志（轮转 + 内存友好）
├── config.py               # 全局配置
├── database.py             # SQLite3 数据库管理
├── model.py                # 数据管线（LRU缓存 / ImageDataset / TextDataset）
├── model_io.py             # 模型存取统一入口（safetensors + 元数据旁车）
├── state.py                # 训练任务状态（内存）
├── cleanup.py              # 存储清理机制
├── resource_limits.py      # 资源占用上限（默认系统一半）
├── requirements.txt        # Python 依赖
├── test_templates/         # 测试页本地示例代码（image/text/diffusion/multimodal 等）
├── .gitcode/               # GitCode CI/CD 流水线
│   ├── workflows/ci.yml    #   编译检查 + 六类训练 E2E 回归
│   └── scripts/            #   E2E 回归驱动脚本
├── architectures/          # 🧱 模型积木库（可插拔）
│   ├── attention.py        #   注意力注册表：full / flash(默认) / linear
│   ├── blocks.py           #   TransformerBlock / 双向编码块(ViT)
│   └── moe.py              #   MoE 专家混合层（修复版）
├── model_zoo/              # 📦 模型定义代码包（按架构分文件存放；注册表见 __init__.py）
│   ├── vision.py           #   CNN(SimpleResNet) / ViT 图像分类
│   ├── text.py             #   Decoder-only 文本生成（积木式注意力+MoE）
│   ├── text_cls.py         #   语言分类（字符级双向编码器 + CLS 池化）
│   ├── diffusion.py        #   扩散生成 DDPM / 扩散编辑适配
│   ├── multimodal.py       #   多模态单流（图文 Decoder-only）
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
| 大语言模型 | Transformer (Decoder-only) | 文本生成：注意力积木可切换 full/flash(默认)/linear，可选 MoE、MLA；支持**拖拽搭建逐层混合注意力**（层数不足自动循环、超出自动截断提醒，可设首尾特殊层）。上传数据支持同级散放 txt 或按文件夹组织 |
| 大语言模型 | Text Classifier | **语言分类**：字符级 Transformer 双向编码器 + CLS 池化分类头，类别由顶层文件夹决定（需按类别文件夹组织数据），同样支持混合注意力 |
| 图像模型 | CNN | 经典卷积分类 |
| 图像模型 | ViT | 视觉 Transformer 分类，支持统一注意力与**拖拽式逐层混合注意力** |
| 图像模型 | Diffusion (DDPM) | 图像生成，噪声预测式扩散；支持 **DDIM 快速采样** |
| 图像模型 | Diffusion Edit Adapter | 图像编辑，条件拼接 + 退化对自监督；DDIM 编辑采样 |
| 多模态 | Single-Stream Decoder | 图文 token 单流拼接，看图续写 |

训练页交互：顶部主条选分区（LLM / 图像 / 多模态）→ 主条下方二级任务条选具体任务
（LLM：文本生成/语言分类；图像：分类/生成/编辑 → 再选子架构 CNN/ViT、DDPM/DDIM、
标准/轻量），各任务的专属参数卡与上传说明自动联动。

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
| GET | `/api/test_template/<name>` | 本地测试示例代码（白名单） |
| GET | `/api/architecture_options` | 积木选项（可用注意力/模型架构/分区） |
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
| GET | `/admin/api/cleanup/status` | 存储占用统计与清理配置 |
| POST | `/admin/api/cleanup/config` | 设置自动清理保留天数 |
| POST | `/admin/api/cleanup/run` | 手动清除模型/上传数据/日志 |
| GET | `/admin/api/logs` | 分页获取行为日志（page/page_size） |
| GET/POST | `/admin/api/log_limits` | 日志存储上限（-1=无上限，默认1000） |
| GET | `/admin/api/resource_limits` | 查看资源占用上限与当前用量 |
| POST | `/admin/api/resource_limits` | 设置资源上限（默认系统一半，0=不限） |

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

本项目基于 **BSD 3-Clause** 许可证开源（见根目录 [`LICENSE`](LICENSE)）。

平台训练产出的所有标准模型导出包（`model.safetensors` + `config.json` + 分词器文件）
均内嵌同一份 BSD-3-Clause `LICENSE` 文件，随模型一起分发。