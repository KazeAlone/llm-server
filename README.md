# llm-server

在 Linux 的 PCIe V100S 32GB 上部署 Qwen3.8-27B。使用 llama.cpp 的 CUDA 后端，提供聊天网页、OpenAI 兼容 API 和 Prometheus 指标。

默认四卡、F16 权重、32K 上下文、单请求槽位。首次执行自动编译 sm_70 后端，将已有的 Safetensors 转成 GGUF，再通过 tmux 后台运行。

## 快速开始

需要 NVIDIA 驱动、CUDA Toolkit 12.x（建议 12.8）、Python 3.10+。用 `nvcc --version` 检查 Toolkit。`nvidia-smi` 顶部的 CUDA Version 是驱动支持上限。为转换准备充足主机内存，建议可用内存 64GB 以上，并在仓库所在磁盘预留约 70GB 输出空间。实际内存需求随模型和转换版本变化。

Ubuntu / Debian 先安装工具。其他发行版使用对应包管理器。

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip libcurl4-openssl-dev curl tmux

git clone https://github.com/KazeAlone/llm-server.git
cd llm-server
cp .env.example .env
nano .env
```

至少修改 `MODEL_DIR`，填现有 18 个权重分片所在目录。目录需包含配置、tokenizer、`model.safetensors.index.json` 和所有分片。

```bash
bash llm.sh doctor
bash llm.sh deploy
bash llm.sh logs
```

`deploy` 会编译、转换并提交后台启动。转换可能持续较长时间，远程执行建议先进入 `tmux new -s llm-deploy`，避免 SSH 中断。后台启动成功提交后，模型仍需要加载，通过以下命令确认就绪。

```bash
bash llm.sh status
curl -f http://127.0.0.1:8080/health
```

输入 `Ctrl+C` 退出日志查看，后台服务继续运行。重复 `deploy` 会复用已有源码和 GGUF，不会自动更新 llama.cpp。首次实际提交记录在 `.runtime/llama-cpp-commit.txt`。更换源模型目录时，应使用新的仓库工作目录，避免误用之前转换的同名 GGUF。一个工作目录内依次执行部署命令。

## 从 Mac 访问

```bash
ssh -N -L 8080:127.0.0.1:8080 用户名@服务器地址
```

浏览器打开 <http://127.0.0.1:8080>。DSH / OpenCode 使用 OpenAI Compatible 类型，Base URL 填 `http://127.0.0.1:8080/v1`，模型名填 `qwen3.8-27b`。客户端要求 API Key 时可填占位值，默认服务通过本机和 SSH 隧道访问。

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好，简短介绍自己。"}],"stream":true,"max_tokens":1024,"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

## 管理命令

| 命令 | 用途 |
| --- | --- |
| `bash llm.sh doctor` | 查看 GPU、拓扑、Toolkit、主机内存和磁盘 |
| `bash llm.sh setup` | 编译后端并安装转换依赖 |
| `bash llm.sh convert` | 转换权重，成功后才发布最终文件 |
| `bash llm.sh start` | 在 tmux 中启动后台服务 |
| `bash llm.sh run` | 前台运行，Ctrl+C 停止 |
| `bash llm.sh status` | 查看后台会话、健康检查和 GPU |
| `bash llm.sh logs` | 查看后台加载和推理日志 |
| `bash llm.sh stop` | 请求停止本目录启动的后台服务 |

tmux 服务持续到退出或主机重启，重启服务器后执行 `bash llm.sh start`。修改运行参数后执行 stop，用 status 确认退出，再 start。日志保存在 `.runtime/server.log`，每次后台启动重新写入。

## 用 topo 选卡

执行 `bash llm.sh doctor`，看 `nvidia-smi topo -m` 中 GPU 行与 GPU 列的交叉位置。

| 标记 | 连接路径 |
| --- | --- |
| `PIX` | 最多经过一个 PCIe 桥 |
| `PXB` | 经过多个 PCIe 桥 |
| `PHB` | 经过 PCIe Host Bridge |
| `NODE` | 同一 NUMA 节点内跨 Host Bridge |
| `SYS` | 跨 NUMA 节点，通过 CPU 间互联 |
| `NV#` | NVLink 连接 |

两卡优先选 PIX / PXB 且空闲显存充足的一组。若 0、1 为一组，2、3 为另一组，组间为 SYS，可以用 `GPU_IDS="0,1"` 或 `GPU_IDS="2,3"`。四卡按 `0,1,2,3` 排序，尽量把同组卡放在一起。实际速度需要测试，拓扑标记不代表保证带宽，四卡按层分配也不会保证四倍生成速度。

`GPU_IDS` 接受 nvidia-smi 的 index 或完整 UUID。脚本先将 index 转为 UUID，再传给 CUDA，进程内部按所填顺序从 CUDA 0 重新编号。无需手动配置 `tensor-split`，脚本按卡数设置相同比例。

## 常用配置

在 `.env` 修改后按需转换、重启。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_DIR` | 必填 | 原始模型目录 |
| `GPU_IDS` | `0,1,2,3` | 指定 GPU 顺序 |
| `PRECISION` | `f16` | 可设 `q8_0`，重新 convert / deploy |
| `VISION` | `0` | 设为 `1` 后 convert，会额外生成视觉模块 |
| `CTX_SIZE` | `32768` | 输入与输出合计容量 |
| `PORT` | `8080` | HTTP 服务端口 |
| `HOST` | `127.0.0.1` | 默认仅本机监听 |
| `FLASH_ATTN` | `auto` | 可改 `off` 排查兼容性 |
| `CUDA_HOME` | 空 | 可填 `/usr/local/cuda-12.8` |
| `BUILD_JOBS` | `8` | 编译并行任务数 |
| `PYTHON_BIN` | `python3` | 创建转换环境的 Python，要求 >=3.10，建议 3.11/3.12 |
| `LLAMA_REF` | 空 | 首次 clone 时选择 tag / commit |

F16 是把 BF16 转为 FP16，均为 16-bit，转换存在舍入。Q8 会减少权重占用。转换在 CPU 上进行，生成文件大小与 GPU 运行显存不同，KV 和计算缓冲会额外占用显存。

## UI 和监控

本脚本直接提供 llama.cpp 内置聊天网页。模型管理后台可以后续接 LocalAI，需使用其自己的后端重新验证兼容性。

```bash
nvidia-smi -l 1
curl http://127.0.0.1:8080/metrics
```

`--metrics` 默认开启，可由 Prometheus 采集。重点看 `llamacpp:prompt_tokens_seconds`、`llamacpp:predicted_tokens_seconds`、`llamacpp:requests_processing`、`llamacpp:requests_deferred`。长期 GPU 历史监控可以增加 DCGM Exporter 和 Grafana。

## 排查

| 问题 | 处理 |
| --- | --- |
| 找不到 nvcc | 安装 Toolkit 或设置 CUDA_HOME |
| pip 找不到 numpy 2.2.6 | 检查 `.runtime/venv/bin/python --version`，NumPy 2.2.6 要求 Python >=3.10；同时检查 pip 镜像和网络 |
| compute_70 不受支持 | 使用 CUDA Toolkit 12.x |
| 缺少分片 / tokenizer | 补全原始模型目录 |
| unknown model architecture | 后端版本可能太旧，使用支持该架构的 llama.cpp 版本 |
| 显存不足 | 先设 CTX_SIZE=8192，检查占用和 GPU_IDS |
| Flash Attention 错误 | 设 FLASH_ATTN=off 后重启 |
| start 后健康检查失败 | 首次加载需要时间，查看 logs 确认原因 |
| CUDA 无可用 kernel | 检查实际二进制是否为本脚本构建的 sm_70 版本 |

本仓库已验证 Shell 语法和模拟启动流程。完整的转换、四卡 GPU 推理、图片与工具调用需在目标服务器验收。源码仓库与 `.env` 不包含模型权重，所有运行产物保存在被 Git 忽略的 `.runtime/`。

如果旧虚拟环境使用 Python 3.8/3.9，安装 Python 3.11 或 3.12 后，在 `.env` 加入 `PYTHON_BIN=python3.11`（也可使用 conda 环境解释器的完整路径）。将 `.runtime/venv` 改名备份，例如 `mv .runtime/venv .runtime/venv-old`，确认备份路径不存在后执行，再重新运行 `bash llm.sh deploy`。仅修改系统 Python 或激活其他环境不会改变已经创建的虚拟环境。

Python 版本符合要求时，可以用 `.runtime/venv/bin/python -m pip download --no-deps --only-binary=:all: --index-url https://pypi.org/simple numpy==2.2.6 -d /tmp/llm-numpy-check` 检查官方源是否提供匹配文件。若官方源成功而原安装失败，检查镜像同步、pip 配置和代理。此命令只下载，完整错误日志仍需结合实际环境判断。

参考 [llama.cpp](https://github.com/ggml-org/llama.cpp)、[NVIDIA 拓扑说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)、[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。
