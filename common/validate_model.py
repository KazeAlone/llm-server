import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
index = root / 'model.safetensors.index.json'
if index.exists():
    shards = set(json.loads(index.read_text())['weight_map'].values())
    for shard in shards:
        path = (root / shard).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            sys.exit(f'缺少或无效权重分片：{shard}')
elif not (root / 'model.safetensors').is_file():
    sys.exit('缺少 model.safetensors 或 model.safetensors.index.json')
