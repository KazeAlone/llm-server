"""Adapt pinned upstream installer to active Conda, no root / Torch replacement.
Fail closed if the upstream recipe changes. CUDA/compiler are host prerequisites.
"""
from pathlib import Path
import re
import sys


def adapt(root):
    text = (root / 'scripts/install_v100.sh').read_text()
    start = text.index('if [[ ${EUID} -eq 0 ]]')
    end = text.index('export CUDA_HOME=/usr/local/cuda-12.8')
    text = text[:start] + text[end:]
    text = text.replace('REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
                        'REPO_ROOT=' + __import__('shlex').quote(str(root)))
    text = text.replace('export CUDA_HOME=/usr/local/cuda-12.8',
                        'export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"')
    text = text.replace('export CUDAHOSTCXX=/usr/bin/g++-12',
                        'export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-12}"')
    # Use this environment's interpreter even if the old tmux PATH differs.
    text = re.sub(r'\bpython\b(?= -| ")', __import__('shlex').quote(sys.executable), text)
    pattern = r'"?' + re.escape(sys.executable) + r'"? -m pip install \\\n  torch==2\.9\.1 torchvision==0\.24\.1 torchaudio==2\.9\.1 \\\n  --index-url https://download\.pytorch\.org/whl/cu128\n'
    text, count = re.subn(pattern, '', text)
    if count != 1:
        raise ValueError('上游 Torch 安装片段发生变化，拒绝运行；需检查适配脚本')
    # Only inference dependencies, not diffusion models.
    text = text.replace('$REPO_ROOT/python[diffusion-v100]', '$REPO_ROOT/python')
    text = text.replace('log "Complete. Run: conda activate sglang-v100"', 'log "Complete. Active Conda environment retained."')
    # Installer still builds the pinned V100 kernels and runs upstream smoke checks.
    if any(token in text for token in ['apt-get ', 'conda activate', 'conda create', 'torch==2.9.1 torchvision']):
        raise ValueError('不允许系统安装、Conda 切换或 Torch 替换片段残留')
    return text


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(adapt(Path(sys.argv[1]).resolve()))
