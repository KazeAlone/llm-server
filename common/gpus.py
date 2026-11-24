"""Resolve nvidia-smi physical indices/UUIDs before exposing GPUs to CUDA."""
import subprocess
import sys


def resolve(spec, inventory):
    rows = [line.split(',') for line in inventory.strip().splitlines()]
    table = {i.strip(): uuid.strip() for i, uuid in rows}
    uuids = list(table.values())
    chosen = []
    for item in spec.split(','):
        item = item.strip()
        if item in table:
            uuid = table[item]
        else:
            matches = [uuid for uuid in uuids if item.startswith('GPU-') and uuid.startswith(item)]
            if len(matches) != 1:
                raise ValueError(f'无效或不唯一 GPU: {item}')
            uuid = matches[0]
        if uuid in chosen:
            raise ValueError(f'重复 GPU: {item}')
        chosen.append(uuid)
    return ','.join(chosen)


if __name__ == '__main__':
    try:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                             '--format=csv,noheader'], text=True)
        print(resolve(sys.argv[1], inventory))
    except (ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
