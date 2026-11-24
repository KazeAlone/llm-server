# llm-server：四套 V100S 部署方式

面向 Linux、4× **PCIe V100S 32GB（sm_70）**、本地 Qwen3.8-27B 权重。每种方式一个目录，各自配置、日志、启停入口；使用 **Conda**，不创建隐藏的 Python venv。

| 目录 | 权重 | 默认 API / UI 端口 | 环境与定位 |
|---|---|---|---|
| [`llama.cpp/`](llama.cpp/) | 原始权重转换 GGUF，F16 / Q8_0 + mmproj | 8080 / 8180 | 保留现有部署，转换优先复用当前 Torch；推理不依赖 Torch |
| [`transformers/`](transformers/) | 直接加载原始 BF16 Safetensors，运行时 FP16 | 8081 / 8181 | 优先复用 Torch 2.10，支持文本、图片、流式 API；作为兼容性基线 |
| [`vllm/`](vllm/) | 同上 | 8082 / 8182 | 固定 v0.19.1 源码编译 sm_70，优先复用 Torch 2.10；V100 实验路线 |
| [`sglang/`](sglang/) | 同上 | 8083 / 8183 | 固定 SGLang-V100 适配源码；该版要求 Torch 2.9.1/cu128，使用独立 Conda 环境 |

**这里没有在真实 V100 上执行编译、加载和测速。** CPU 测试覆盖脚本参数、选卡、API 鉴权、图片路径限制、流式返回/异常和 UI 代理。vLLM/SGLang 的 V100 路线需要本机实测，不能保证超过现有 llama.cpp，更不能照搬 NVLink 机器的速度。

## 已经部署过：直接拉取，不覆盖原配置

```bash
git pull --ff-only
bash llm.sh status
```

原来的根目录 `.env`、`.runtime`、GGUF、mmproj、编译产物、tmux 会话都继续使用，根目录 `bash llm.sh ...` 仍指向 llama.cpp。**不需要重新复制 `.env`、转换模型或重建后端。**

新入口 `bash llama.cpp/llm.sh ...` 与根入口控制同一个服务。llama.cpp 优先读取 `llama.cpp/.env`，没有才读取旧根目录 `.env`；已有用户可继续保留根配置。新三套方式不自动读取根配置，避免把 GGUF 参数误传给其他后端。

## 先试 Transformers，再试 vLLM / SGLang

先停止占卡的旧服务，不要同时加载多个 27B FP16 实例。后台安装本身不等于模型已加载，但环境自检和部分编译会访问 GPU。

```bash
# 仓库根目录；先激活你已有的 Torch 2.10 环境
conda activate 你的环境名
bash llm.sh stop

cd transformers
# 仅第一次复制，之后只编辑已有 .env
cp -n .env.example .env
nano .env
```

至少改这些，**MODEL_DIR 指向含 config.json 和 18 个原始分片的目录，不是 GGUF 文件**：

```bash
MODEL_DIR="/data/models/Qwen3.8-27B"
GPU_IDS="0,1,2,3"
HOST="0.0.0.0"
PORT=8081
API_KEY="自己设置一个长密钥"
CTX_SIZE=8192
VISION=1
MEDIA_PATH="/"
CUDA_HOME="/usr/local/cuda-12.8"
```

```bash
bash llm.sh doctor
bash llm.sh setup      # 安装依赖；Transformers 不转换权重
bash llm.sh start      # tmux 后台启动
bash llm.sh logs       # 等到模型加载完成；Ctrl+C 只退出日志
```

`setup` 使用当前 `CONDA_PREFIX/bin/python`，先验证 CUDA 与 sm_70，再把已有 Torch / torchvision / torchaudio / Triton 版本写入约束。后续 pip 包括构建子进程继承约束；**冲突时退出，不自动换成 Torch 2.11**。已有 Torch 必须本身包含 V100 CUDA 支持，只有版本号相同还不够。

建议给三个 Python 后端各自建环境，复用版本、隔离其余依赖：

```bash
conda create -n llm-transformers --clone 你的原环境名
conda create -n llm-vllm --clone 你的原环境名
```

克隆会复制已安装的 Torch，不重新下载 Torch。不同环境占用更多磁盘，但避免互相改动 Transformers、Triton 等依赖。也可直接使用现有环境，需接受其他 Python 依赖会被安装/调整。

SGLang 的专用环境步骤见 [sglang/README.md](sglang/README.md)，它不能按本适配版直接复用 2.10。

## 公共命令与配置

在相应目录执行：

| 命令 | 用途 |
|---|---|
| `bash llm.sh doctor` | Python/Torch/CUDA、GPU 列表、`nvidia-smi topo -m` |
| `bash llm.sh setup` | 安装或源码编译，不加载模型；重复执行复用源码 |
| `bash llm.sh start` | 后台启动；使用 setup 记录的 Conda 解释器 |
| `bash llm.sh run` | 前台启动，便于查看完整报错 |
| `bash llm.sh restart` | 修改 `.env` 后重启；无需再复制 env |
| `bash llm.sh stop` | 向本目录服务的真实 pane ID 发送 Ctrl+C，最多等待 30 秒 |
| `bash llm.sh status` | tmux 状态 + HTTP `/health`；加载中可能尚未通过 |
| `bash llm.sh logs` | 本后端日志 |
| `bash llm.sh diagnose` | GPU 信息 + 最近日志 |
| `bash llm.sh ui` | 前台启动共用的网页聊天/监控服务 |

`deploy` = setup + start（llama.cpp 还包含转换）。改密钥、IP、选卡、上下文后用 `restart`；改依赖或源码版本才需 setup。setup/start/stop/restart 有目录级互斥锁，避免重复操作。

新三套环境的状态放各自 `.runtime/`。`.env`、运行产物、权重不会提交到 Git。密码含 `$` 等 Shell 字符时使用单引号：`API_KEY='字面量密钥'`。`.env` 是本机 Shell 配置文件，可以用数组 `EXTRA_ARGS=(--某参数 某值)`，不要粘贴不可信脚本。

## 网页聊天、图片与监控

模型服务启动后，在同一部署目录另开终端：

```bash
conda activate 对应环境
bash llm.sh ui
```

例如 Transformers 浏览器打开 `http://127.0.0.1:8181`。输入与 `.env` 一致的 API 密钥，即可流式聊天、上传图片、刷新健康状态、GPU 显存/占用/功耗和原始 Prometheus 指标。页面密钥仅保存在页面内存，不写入浏览器存储。默认 UI 仅本机访问；远程可 SSH 转发，或在 `.env` 设置 `UI_HOST="0.0.0.0"` 后重新运行 UI。API 与 UI 是两个独立进程，停模型不会关闭 UI；UI 用 Ctrl+C 关闭。

llama.cpp 自带的 Web UI 仍可直接使用。新增共用 UI 通过同源代理访问 API，不要求后端打开 CORS。监控显示服务器全部 GPU，便于检查残留进程；原始指标中的 token 统计/时延含义由各后端定义，不能直接把累计吞吐当单请求 decode 速度。

## API 接入

四个后端统一 `MODEL_ID=qwen3.8-27b`（llama.cpp 原有固定别名也是这个）。客户端填写：

```text
Base URL: http://服务器局域网IP:后端端口/v1
API key:  .env 中的 API_KEY
Model:    qwen3.8-27b
```

`HOST=0.0.0.0` 是服务监听地址；客户端必须填服务器实际 IP。不同目录默认端口不同，旧 `.env` 中自定义端口保持原值。

```bash
# 以 Transformers 为例；不要把密钥写进仓库
export TEST_API_KEY='你的密钥'
curl http://127.0.0.1:8081/v1/chat/completions \
  -H "Authorization: Bearer $TEST_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好"}],"max_tokens":256,"stream":true}'
```

图片消息采用 OpenAI `image_url` 格式：

```json
{"model":"qwen3.8-27b","messages":[{"role":"user","content":[{"type":"text","text":"描述这张图片"},{"type":"image_url","image_url":{"url":"file:///data/images/test.jpg"}}]}],"max_tokens":256}
```

这是**服务器本机文件**。客户端电脑的图片使用网页上传（base64）或客户端自行转 data URL。Transformers、vLLM、SGLang 不需要 mmproj；只有 llama.cpp 需要 `VISION=1` 并转换 mmproj。

`MEDIA_PATH="/"` 延续之前允许整个文件系统的设置；可收窄为 `/data/images`，空字符串禁用本地 `file://`。SGLang 的限制通过本仓库 HTTP 中间件作用于 `/v1/chat/completions`、`/generate` 的本地媒体输入；不覆盖其他原生管理/文件接口。生产使用建议只暴露 OpenAI 接口。Transformers 基线不支持工具调用、并发批处理、视频/音频输入；Agent 工具调用用另外三个后端。

## 选卡与性能比较

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv
nvidia-smi topo -m
```

`GPU_IDS` 填 **nvidia-smi 的物理编号**或 UUID，以你的拓扑矩阵选择。两卡优先选择 `PIX`/`PXB`、再比较 `PHB`，跨 CPU 的 `SYS` 通常通信代价更大；同时确认空闲显存。不要默认编号相邻就最近。本仓库会将编号转成 UUID 再设置 CUDA_VISIBLE_DEVICES，避免编号重排。

27B FP16 权重本身约 50 多 GiB，**单张 32GB 放不下**；新三套默认四卡，建议先成功再按拓扑比较两卡。单卡量化继续用现有 llama.cpp Q8_0，能否完整驻留需同时考虑视觉模块、KV cache 和上下文预算。

Transformers 的 `device_map=auto` 是层分配，多卡并不等于每层张量并行，通常不适合作为提速首选。vLLM/SGLang 使用 TP，PCIe 通信可能影响延迟；默认关闭自定义 all-reduce，不启用 NVLink/NVLS 专用参数，也不强制 `NCCL_P2P_LEVEL=NVL`。先测 FP16 正确性，再考虑特殊量化或投机解码。

对比时固定相同输入、输出长度、思考设置、图片分辨率和上下文，预热后重复几次。分别记录 TTFT、单请求 decode tokens/s、多人总吞吐。SGLang 的 NVLink / FP8 / DFlash 测试值不代表本脚本的 PCIe / FP16 默认配置。

## 系统依赖 / 排错

需要已有 NVIDIA 驱动、对应 CUDA 12.x Toolkit（vLLM 12.6–12.8，SGLang 12.8）、C/C++ 编译器、git、tmux、flock；新 Python 后端要求 Python >=3.11，建议 3.12。不要求 Docker。

Ubuntu/Debian 安装基础工具（按服务器权限自行执行一次）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git tmux util-linux curl cmake ninja-build g++-12
```

- Conda 不会接管 `pip` 的解析器；脚本使用当前环境的 `python -m pip`，并额外限制现有 Torch 版本。`ResolutionImpossible` 时保留报错，不要去掉约束强装。
- `couldn't bind`：检查 `ss -ltnp`、HOST 是否属于本机、端口是否冲突。四套虽端口不同，仍共享 GPU 显存。
- `no kernel image` / `sm_70`：检查 Torch 架构列表和本地编译结果。不要直接换成 CUDA13 或最新版预编译推理轮子。
- 更改源码 REF 后，已有 checkout 版本不匹配会退出；保留旧 `.runtime/source` 并改名，再 setup。不会自动覆盖自定义源码改动。
- `start` 返回仅表示创建了后台会话；必须查看 logs/status 确认模型加载成功。首轮 JIT 可能较久。
- 原 llama.cpp 的转换、q8、mmproj、CUDA 构建细节见 [旧部署说明](llama.cpp/README.md)。

## 来源与验证

- [Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
- [vLLM v0.19.1 源码](https://github.com/vllm-project/vllm/tree/b1388b1fbf5aaef47937fabe98931211684666a6)、[既有 Torch 构建说明](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)
- [V100 vLLM 社区复现](https://github.com/KumphanartDansiri/v100-vllm-2026)（其他模型/拓扑的复现证据，不是本项目硬件验收）
- [SGLang-V100 固定版本](https://github.com/haohervchb/sglang-V100/tree/d85b3e2705411df781dadf9e8fa4db52c8f07a5a)，保留其上游依赖补丁、sm_70 构建和 smoke checks；本项目不复制其内核源码。

本地无 GPU 测试：

```bash
python -m pip install pytest fastapi httpx requests prometheus-client pillow uvicorn
python -m pytest -q tests
```

测试中的 Torch/Transformers 模型是 mock；不会下载 27B 模型，也不能证明 GPU 内核兼容或模型回答正确。
