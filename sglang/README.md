# SGLang：专用 V100 适配版（实验）

使用 [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) 固定 `d85b3e27`，包含 sm_70 内核及 Qwen3.8 适配。其当前安装配方和 GPU smoke checks 固定 **Torch2.9.1 + CUDA12.8**。因此本后端不强行复用或降级你的 Torch2.10。

单独创建环境，只在该环境安装匹配栈：

```bash
conda create -n llm-sglang python=3.12 pip -y
conda activate llm-sglang
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

cd sglang
cp -n .env.example .env
nano .env
# 填 MODEL_DIR，CUDA_HOME 指向 /usr/local/cuda-12.8
bash llm.sh doctor
bash llm.sh setup
bash llm.sh start
bash llm.sh logs
```

需要事先具备 CUDA12.8 Toolkit、g++12、cmake、ninja、git 等系统工具。脚本**不自动运行 sudo/apt、不安装或切换 Conda、不替换 Torch**。如果激活的是2.10环境，setup 会在编译前退出并给出专用环境指引。

setup 对固定上游安装配方生成本地适配脚本，移除系统安装、Conda 和重装 Torch 段，只安装推理包（不安装 diffusion extras）；保留原依赖版本、补丁、FlashInfer SM70、TurboMind、sglang-kernel、Marlin 构建及 GPU smoke tests。生成前严格核对关键片段，匹配失败就停止。部分上游依赖源码位于本目录 `.runtime/deps`；上游 CUTLASS 还会使用 `~/cutlass`。不会因为主模型是稠密就跳过 smoke 要求的内核。

这个版本的上游 smoke 会核对 NCCL2.27.5，因此安装配方会在**当前专用环境**恢复其所需的 CUDA12 NCCL。不要把它与 vLLM 共用环境。默认 `BUILD_JOBS=4`；编译时内存不足可降到2。实际编译时间与机器有关。

默认原始权重 FP16、四卡 TP、TileLang V100 语言注意力、Triton 视觉注意力；不启用 NVLS、不强制 NVLink 通信级别、关闭自定义 all-reduce。图像无需 mmproj，支持密钥、Qwen工具调用/思考解析、指标及共用 UI。VISION=0 时 HTTP 适配层拒绝图片消息。

保留官方入口的 plugin 加载和子进程清理。额外用 HTTP 适配层检查 `/v1/chat/completions`、`/generate` 中本地图片路径；`MEDIA_PATH` 为空禁用 file://，为 `/` 允许服务器整个文件系统。该中间件不替代 SGLang 所有原生接口的权限管理。

不默认启用 FP8 KV、FP8 模型、MTP/DFlash 或 FA-V100 视觉额外插件，先测试当前 BF16 分片以 FP16 正确运行。上游部分高速度结果使用 NVLink、FP8、投机解码，不能当作本项目四张 PCIe V100S 的预期速度。**本仓库未在真实 V100 上加载验证。**

通用启停、UI、API、拓扑选卡参见 [主说明](../README.md)。
