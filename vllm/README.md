# vLLM：V100 源码路线（实验）

固定 **v0.19.1 / b1388b1**，它的 CUDA12 构建列表仍包含 sm_70。使用官方 `use_existing_torch.py --prefix` 移除 Torch 系列安装钉死项，按现有版本约束安装构建依赖，以 `TORCH_CUDA_ARCH_LIST=7.0`、`--no-build-isolation` 编译。**不是直接 pip install 最新 vllm**。

```bash
conda create -n llm-vllm --clone 原Torch环境名
conda activate llm-vllm
cd vllm
cp -n .env.example .env
nano .env
bash llm.sh doctor
bash llm.sh setup
bash llm.sh start
bash llm.sh logs
```

优先使用已有 Torch 2.10 CUDA12.x；设置 CUDA_HOME 指向 12.6–12.8 Toolkit。版本约束冲突时会退出，不自动升级 Torch。源码构建需要较多 RAM、磁盘和时间，默认 `BUILD_JOBS=4`。

默认四卡 FP16 TP，语言注意力 TRITON_ATTN，视觉编码器 TORCH_SDPA；关闭自定义 all-reduce，启用分块 prefill。支持原生 OpenAI 文本/图片 API、Qwen reasoning/tool-call parser、指标。密钥用 VLLM_API_KEY 环境变量，避免落到进程参数。选卡、上下文、图片路径和 UI 见 [主说明](../README.md)。

如果遇到图捕获问题，可改 `ENFORCE_EAGER=1` 验证正确性，之后再比较速度。原始 BF16 权重加载为 FP16；没有接入社区 FP8 反量化插件，不可把 FP8 权重直接当默认方案。

v0.19.1 的 sm_70 构建路径与社区同家族模型复现可作为起点，**本仓库没有完成 Qwen3.8-27B + PCIe V100 的真实 GPU 验收**。模型、视觉或混合注意力内核仍可能需要适配；先保留失败日志，再调整版本/补丁。不要仅凭安装成功认为图片链路和生成正确性均通过。
