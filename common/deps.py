"""Run pip without replacing an existing Torch stack (also in child pip calls)."""
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys


def constraints(destination):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('Torch 无法使用 CUDA；检查驱动、CUDA wheel 和 V100 sm_70 支持。')
    if 'sm_70' not in torch.cuda.get_arch_list():
        raise SystemExit('当前 Torch 未编译 sm_70，不能用于 V100。请安装含 sm_70 的 CUDA 12.x wheel。')
    packages = ['torch', 'torchvision', 'torchaudio', 'triton']
    lines = []
    for package in packages:
        try:
            lines.append(f'{package}=={metadata.version(package)}')
        except metadata.PackageNotFoundError:
            pass
    Path(destination).write_text('\n'.join(lines) + '\n')
    print('锁定已有依赖：' + ', '.join(lines), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'lock':
        constraints(sys.argv[2])
    else:
        env = os.environ.copy()
        subprocess.run([sys.executable, '-m', 'pip', 'install', *sys.argv[1:]],
                       env=env, check=True)
