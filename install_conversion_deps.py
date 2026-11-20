"""Install conversion dependencies while retaining a working existing PyTorch."""

import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    requirements = Path(sys.argv[1]).resolve()
    command = [sys.executable, "-m", "pip", "install"]
    if importlib.util.find_spec("torch") is None:
        print("当前环境没有 Torch，按上游转换依赖安装。", flush=True)
        subprocess.run(command + ["-r", str(requirements)], check=True)
        return

    # An import failure should surface before pip changes this environment.
    import torch

    version = torch.__version__
    print(f"复用当前 Torch {version}，路径 {torch.__file__}", flush=True)
    lines = requirements.read_text().splitlines(keepends=True)
    filtered = [line for line in lines if not re.match(r"^\s*torch\s*(?:[<>=!~;\[]|$)", line)]

    # Keep relative -r includes relative to the original requirements directory.
    with tempfile.TemporaryDirectory(prefix="llm-deps-", dir=requirements.parent) as directory:
        directory = Path(directory)
        override = directory / "requirements.txt"
        constraints = directory / "constraints.txt"
        rewritten = []
        for line in filtered:
            if line.lstrip().startswith("-r "):
                include = line.strip()[3:].strip()
                rewritten.append(f"-r {requirements.parent / include}\n")
            else:
                rewritten.append(line)
        override.write_text("".join(rewritten))
        constraints.write_text(f"torch=={version}\n")
        subprocess.run(command + ["-r", str(override), "-c", str(constraints)], check=True)


if __name__ == "__main__":
    main()
