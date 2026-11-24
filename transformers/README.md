# Transformers

使用当前 Conda/Torch，直接读取原始 HF 分片，运行时转 FP16。`device_map=auto` 分配到选定 GPU，不是 TP；支持 OpenAI 文本/图片 chat API、真实流式输出、密钥、模型列表、健康检查和 Prometheus 指标。

```bash
conda activate llm-transformers
cd transformers
cp -n .env.example .env
nano .env
bash llm.sh doctor
bash llm.sh setup
bash llm.sh start
bash llm.sh logs
# 另一个终端，同目录、同环境
bash llm.sh ui
```

环境可以提前 `conda create -n llm-transformers --clone 原环境名`，继承已有 Torch 2.10。详细配置、API 和迁移方法见 [仓库说明](../README.md)。

默认 `ATTN_IMPLEMENTATION=sdpa`，V100 下可使用 PyTorch math fallback；不安装要求新架构的 FlashAttention 2。遇到具体 attention 兼容问题可改为 `eager` 再 restart。混合线性注意力缺少融合内核时，Transformers 的通用实现可能较慢，这套主要用于确认原始模型和图片链路。

`GPU_MEMORY_GIB=28` 限制每张可用于自动分配的显存；不允许静默 CPU/disk offload。OOM 时先降低上下文/图片分辨率，检查其他 GPU 进程，再考虑调整预算。四卡默认加载原始 FP16，不做额外量化。

这是单请求基线：忙时返回429。支持 `/v1/models`、`/v1/chat/completions`、`/health`、`/metrics`，支持 `stream`、`max_tokens`/`max_completion_tokens`、temperature、top_p、chat_template_kwargs；不实现完整 OpenAI API，不支持工具调用、视频和并发批处理。思考文字按模型解码结果输出，不单独拆分 reasoning 字段。
