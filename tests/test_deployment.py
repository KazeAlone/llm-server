import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import types

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_mapping():
    mod = load(ROOT / 'common/gpus.py', 'gpu_helper')
    inventory = '0, GPU-aaaa\n1, GPU-bbbb\n2, GPU-cccc\n'
    assert mod.resolve('2,0', inventory) == 'GPU-cccc,GPU-aaaa'
    assert mod.resolve('GPU-bb', inventory) == 'GPU-bbbb'
    for value in ('0,0', '5', '', '0,,1'):
        with pytest.raises(ValueError):
            mod.resolve(value, inventory)


def test_missing_shard(tmp_path):
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'missing.safetensors'}}))
    result = subprocess.run([sys.executable, str(ROOT / 'common/validate_model.py'), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and 'missing.safetensors' in result.stderr


def test_sglang_local_media(tmp_path, monkeypatch):
    mod = load(ROOT / 'sglang/launch.py', 'sglang_launcher')
    monkeypatch.setenv('MEDIA_PATH', str(tmp_path))
    mod.check_media({'image_url': {'url': f'file://{tmp_path}/a.png'}})
    with pytest.raises(ValueError):
        mod.check_media({'image_data': '/etc/passwd'})
    monkeypatch.setenv('MEDIA_PATH', '')
    with pytest.raises(ValueError):
        mod.check_media({'image_url': {'url': f'file://{tmp_path}/a.png'}})


def test_upstream_installer_guard(tmp_path):
    mod = load(ROOT / 'sglang/prepare_install.py', 'installer_adapter')
    (tmp_path / 'scripts').mkdir()
    script = '''REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${EUID} -eq 0 ]]; then
sudo apt-get update
conda create -n whatever
export CUDA_HOME=/usr/local/cuda-12.8
python -m pip install \\
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \\
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$REPO_ROOT/python[diffusion-v100]"
log "Complete. Run: conda activate sglang-v100"
'''
    file = tmp_path / 'scripts/install_v100.sh'
    file.write_text(script)
    adapted = mod.adapt(tmp_path)
    assert 'apt-get' not in adapted and 'conda create' not in adapted
    assert 'torch==2.9.1 torchvision' not in adapted
    assert '[diffusion-v100]' not in adapted
    file.write_text(script.replace('torch==2.9.1', 'torch==2.10.0'))
    with pytest.raises(ValueError):
        mod.adapt(tmp_path)


@pytest.fixture(scope='module')
def api():
    fake_torch = types.ModuleType('torch')
    from contextlib import nullcontext
    fake_torch.inference_mode = nullcontext
    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoModelForImageTextToText = object
    fake_transformers.AutoProcessor = object
    fake_transformers.StoppingCriteria = object

    class Streamer:
        def __init__(self, *args, **kwargs):
            self.queue = queue.Queue()
        def __next__(self):
            value = self.queue.get(timeout=0.05)
            if value is None:
                raise StopIteration
            return value
    fake_transformers.TextIteratorStreamer = Streamer
    old = {key: sys.modules.get(key) for key in ('torch', 'transformers')}
    sys.modules.update(torch=fake_torch, transformers=fake_transformers)
    mod = load(ROOT / 'transformers/server.py', 'transformers_server_test')

    class Tensor:
        shape = (1, 3)
    class Inputs(dict):
        def to(self, device):
            return self
    class Processor:
        tokenizer = None
        def apply_chat_template(self, *args, **kwargs):
            return Inputs(input_ids=Tensor())
        def decode(self, *args, **kwargs):
            return '你好'
    class Generated:
        shape = (1, 5)
        def __getitem__(self, item):
            return [1, 2]
    class Model:
        device = 'cuda:0'
        fail = False
        def generate(self, **kwargs):
            if self.fail:
                raise RuntimeError('mock GPU failure')
            if 'streamer' in kwargs:
                kwargs['streamer'].queue.put('你好')
                kwargs['streamer'].queue.put(None)
            return Generated()
    mod.processor, mod.model = Processor(), Model()
    yield mod
    for key, value in old.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


def payload(**kwargs):
    return dict(model='qwen3.8-27b', messages=[{'role': 'user', 'content': '你好'}], **kwargs)


def test_auth_and_chat(api, monkeypatch):
    client = TestClient(api.app)
    monkeypatch.setattr(api, 'API_KEY', 'secret')
    assert client.get('/v1/models').status_code == 401
    headers = {'Authorization': 'Bearer secret'}
    response = client.post('/v1/chat/completions', headers=headers, json=payload())
    assert response.status_code == 200
    assert response.json()['usage']['completion_tokens'] == 2
    assert response.json()['choices'][0]['message']['content'] == '你好'
    assert not api.lock.locked()


def test_stream_success_and_failure(api, monkeypatch):
    monkeypatch.setattr(api, 'API_KEY', '')
    client = TestClient(api.app)
    response = client.post('/v1/chat/completions', json=payload(stream=True))
    assert '你好' in response.text and '[DONE]' in response.text
    monkeypatch.setattr(api.model, 'fail', True)
    response = client.post('/v1/chat/completions', json=payload(stream=True))
    assert 'mock GPU failure' in response.text and '[DONE]' in response.text
    assert not api.lock.locked()


def test_context_and_busy(api, monkeypatch):
    monkeypatch.setattr(api, 'API_KEY', '')
    client = TestClient(api.app)
    assert client.post('/v1/chat/completions', json=payload(max_tokens=999999)).status_code == 400
    assert not api.lock.locked()
    api.lock.acquire()
    try:
        assert client.post('/v1/chat/completions', json=payload()).status_code == 429
    finally:
        api.lock.release()
    assert client.post('/v1/chat/completions', json=payload(tools=[{}])).status_code == 400


def test_image_allowlist(api, tmp_path, monkeypatch):
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (2, 2)).save(image)
    monkeypatch.setenv('MEDIA_PATH', str(tmp_path))
    monkeypatch.setenv('VISION', '1')
    assert api.load_image(image.as_uri()).size == (2, 2)
    with pytest.raises(ValueError):
        api.load_image('file:///etc/passwd')
    monkeypatch.setenv('MEDIA_PATH', '')
    with pytest.raises(ValueError):
        api.load_image(image.as_uri())


@pytest.mark.parametrize('backend', ['transformers', 'vllm', 'sglang'])
def test_launch_command(backend, tmp_path):
    repo = tmp_path / 'repo'
    shutil.copytree(ROOT / 'common', repo / 'common', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / backend, repo / backend, ignore=shutil.ignore_patterns('.env', '.runtime', '__pycache__'))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').touch()
    envdir = tmp_path / 'conda'
    (envdir / 'bin').mkdir(parents=True)
    fake = envdir / 'bin/python'
    fake.write_text(f'''#!{sys.executable}
import json, os, sys, subprocess
if sys.argv[1].endswith('gpus.py'):
 print('GPU-test0,GPU-test1')
elif sys.argv[1].endswith('validate_model.py'):
 subprocess.run([{sys.executable!r}, *sys.argv[1:]], check=True)
else:
 print(json.dumps({{'argv':sys.argv[1:], 'gpus':os.environ.get('CUDA_VISIBLE_DEVICES'), 'key':os.environ.get('VLLM_API_KEY', os.environ.get('LLM_SERVER_API_KEY', os.environ.get('API_KEY')))}}))
''')
    fake.chmod(0o755)
    env = dict(os.environ, CONDA_PREFIX=str(envdir), MODEL_DIR=str(model), GPU_IDS='0,1', API_KEY='test-secret')
    run = subprocess.run(['bash', str(repo / backend / 'llm.sh'), 'run'], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result['gpus'] == 'GPU-test0,GPU-test1'
    assert result['key'] == 'test-secret'
    assert 'test-secret' not in ' '.join(result['argv'])
    if backend in ('vllm', 'sglang'):
        assert '--disable-custom-all-reduce' in result['argv']
        assert '--enable-nccl-nvls' not in result['argv']
        assert result['argv'][result['argv'].index('--tensor-parallel-size') + 1] == '2'


def test_torch_stack_lock(tmp_path, monkeypatch):
    mod = load(ROOT / 'common/deps.py', 'deps_helper')
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, get_arch_list=lambda: ['sm_70']))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    versions = dict(torch='2.10.0+cu128', torchvision='0.25.0', torchaudio='2.10.0', triton='3.6.0')
    monkeypatch.setattr(mod.metadata, 'version', versions.__getitem__)
    file = tmp_path / 'constraints.txt'
    mod.constraints(file)
    assert file.read_text().splitlines() == [f'{p}=={v}' for p, v in versions.items()]
    torch.cuda.get_arch_list = lambda: ['sm_80']
    with pytest.raises(SystemExit, match='sm_70'):
        mod.constraints(file)


def test_legacy_entry_config_precedence(tmp_path):
    (tmp_path / 'llama.cpp').mkdir()
    shutil.copy(ROOT / 'llm.sh', tmp_path / 'llm.sh')
    shutil.copy(ROOT / 'llama.cpp/llm.sh', tmp_path / 'llama.cpp/llm.sh')
    (tmp_path / '.env').write_text('PRECISION=invalid\n')
    result = subprocess.run(['bash', str(tmp_path / 'llm.sh'), 'help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'PRECISION' in result.stderr
    (tmp_path / 'llama.cpp/.env').write_text('PRECISION=f16\n')
    result = subprocess.run(['bash', str(tmp_path / 'llm.sh'), 'help'], capture_output=True, text=True)
    assert result.returncode == 0
    nested = subprocess.run(['bash', str(tmp_path / 'llama.cpp/llm.sh'), 'help'], capture_output=True, text=True)
    assert result.stdout == nested.stdout


def test_stop_uses_actual_pane(tmp_path):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log, stopped = tmp_path / 'log', tmp_path / 'stopped'
    tmux = bindir / 'tmux'
    tmux.write_text(f'''#!/usr/bin/env bash
printf '%s\\n' "$*" >> '{log}'
case $1 in
 has-session) [[ ! -f '{stopped}' ]] ;;
 list-panes) printf '%%42\\n' ;;
 send-keys) touch '{stopped}' ;;
 *) exit 1 ;;
esac
''')
    tmux.chmod(0o755)
    script = f'''BACKEND_DIR={ROOT / 'vllm'}
BACKEND=vllm DEFAULT_PORT=8082
source '{ROOT / 'common/runtime.sh'}'
stop
'''
    result = subprocess.run(['bash', '-c', script], env=dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}'), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'send-keys -t %42 C-c' in log.read_text()


def test_console_proxy(tmp_path):
    import http.server
    import socket
    import time
    import urllib.request
    import urllib.error

    class Backend(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.headers['Authorization'] == 'Bearer test-secret'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.write(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
        def log_message(self, *args):
            pass

    backend = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1', 0))
        port = port_socket.getsockname()[1]
    process = subprocess.Popen([sys.executable, str(ROOT / 'common/console.py'), '--port', str(port),
                                '--upstream', f'http://127.0.0.1:{backend.server_port}'],
                               env=dict(os.environ, API_KEY='test-secret'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f'http://127.0.0.1:{port}'
    try:
        for attempt in range(60):
            try:
                with urllib.request.urlopen(base + '/', timeout=1) as response:
                    assert b'test-secret' not in response.read()
                break
            except urllib.error.URLError:
                if process.poll() is not None:
                    pytest.fail('console exited')
                time.sleep(0.05)
        else:
            pytest.fail('console did not start')
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(base + '/health')
        assert denied.value.code == 401
        headers = {'Authorization': 'Bearer test-secret', 'Content-Type': 'application/json'}
        with urllib.request.urlopen(urllib.request.Request(base + '/health', headers=headers)) as response:
            assert json.load(response)['status'] == 'ok'
        with urllib.request.urlopen(urllib.request.Request(base + '/v1/chat/completions', headers=headers, data=b'{}')) as response:
            assert b'[DONE]' in response.read()
        backend.shutdown()
        backend.server_close()
        with pytest.raises(urllib.error.HTTPError) as down:
            urllib.request.urlopen(urllib.request.Request(base + '/health', headers=headers))
        assert down.value.code == 502
    finally:
        process.terminate()
        process.wait(timeout=5)
        backend.shutdown()
        backend.server_close()
