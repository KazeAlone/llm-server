"""Small, serialized Transformers baseline with OpenAI text/image chat API."""
import base64
from contextlib import asynccontextmanager
from io import BytesIO
import json
import os
from pathlib import Path
from queue import Empty
import secrets
import threading
import time
from urllib.parse import unquote, urlparse
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
import requests
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer, StoppingCriteria
import uvicorn

MODEL_ID = os.environ.get('MODEL_ID', 'qwen3.8-27b')
CTX_SIZE = int(os.environ.get('CTX_SIZE', '8192'))
API_KEY = os.environ.get('API_KEY', '')
lock = threading.Lock()
model = processor = None
REQUESTS = Counter('llm_completed_requests', 'Successful requests')
TOKENS = Counter('llm_generated_tokens', 'Completion tokens')
LATENCY = Histogram('llm_request_seconds', 'Inference duration')


@asynccontextmanager
async def lifespan(app):
    global model, processor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    path = os.environ['MODEL_DIR']
    count = torch.cuda.device_count()
    memory = {i: f"{int(os.environ.get('GPU_MEMORY_GIB', '28'))}GiB" for i in range(count)}
    memory['cpu'] = '0GiB'
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        path, local_files_only=True, dtype=torch.float16, device_map='auto',
        max_memory=memory, attn_implementation=os.environ.get('ATTN_IMPLEMENTATION', 'sdpa'))
    if any(str(value) in ('cpu', 'disk') for value in model.hf_device_map.values()):
        raise RuntimeError('模型无法全部驻留 GPU；增加选卡或显存预算。此基线不允许静默 CPU offload。')
    model.eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware('http')
async def authenticate(request: Request, call_next):
    supplied = request.headers.get('authorization', '')
    if API_KEY and not secrets.compare_digest(supplied, f'Bearer {API_KEY}'):
        return Response(content='{"error":{"message":"Unauthorized"}}', status_code=401,
                        media_type='application/json')
    return await call_next(request)


class Chat(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    max_tokens: int = Field(default=512, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    tools: list[dict] | None = None
    n: int = 1
    chat_template_kwargs: dict = Field(default_factory=dict)


def load_image(url):
    if os.environ.get('VISION', '1') != '1':
        raise ValueError('VISION=0，图片输入已关闭')
    parsed = urlparse(url)
    if parsed.scheme == 'file':
        allowed = os.environ.get('MEDIA_PATH', '/')
        if not allowed:
            raise ValueError('本地图片访问已禁用')
        if parsed.netloc not in ('', 'localhost'):
            raise ValueError('file URL 不能包含远程主机名')
        path = Path(unquote(parsed.path)).resolve()
        if not path.is_relative_to(Path(allowed).resolve()):
            raise ValueError('图片位于 MEDIA_PATH 之外')
        image = Image.open(path)
    elif url.startswith('data:image/'):
        image = Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1], validate=True)))
    elif parsed.scheme in ('http', 'https'):
        with requests.get(url, timeout=30, stream=True) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > 20 * 1024 * 1024:
                    raise ValueError('远程图片超过 20MiB')
        image = Image.open(BytesIO(data))
    else:
        raise ValueError('图片 URL 需要 http(s)、data:image 或 file:///absolute/path')
    with image:
        return image.convert('RGB')


def normalize(messages):
    result = []
    for message in messages:
        if message.get('role') not in ('system', 'user', 'assistant'):
            raise ValueError('基线仅支持 system/user/assistant 消息')
        content = message.get('content', '')
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        normalized = []
        for part in content:
            if part.get('type') == 'text':
                normalized.append({'type': 'text', 'text': part['text']})
            elif part.get('type') == 'image_url':
                value = part['image_url']
                url = value['url'] if isinstance(value, dict) else value
                normalized.append({'type': 'image', 'image': load_image(url)})
            else:
                raise ValueError('仅支持 text 和 image_url 内容')
        result.append({'role': message['role'], 'content': normalized})
    if not result:
        raise ValueError('messages 不能为空')
    return result


@app.get('/health')
def health():
    if model is None:
        raise HTTPException(503, 'loading')
    return {'status': 'ok'}


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL_ID, 'object': 'model', 'owned_by': 'local'}]}


@app.get('/metrics')
def metrics():
    return Response(generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


class Cancelled(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, *args, **kwargs):
        return self.event.is_set()


@app.post('/v1/chat/completions')
def chat(body: Chat):
    if body.model != MODEL_ID:
        raise HTTPException(404, '未知 model id')
    if body.tools or body.n != 1:
        raise HTTPException(400, 'Transformers 基线不支持工具调用或 n!=1；Agent 请用其他三个后端')
    if not lock.acquire(blocking=False):
        raise HTTPException(429, 'GPU 正在生成，请稍后重试')
    try:
        messages = normalize(body.messages)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt', **body.chat_template_kwargs)
        inputs = inputs.to(model.device)
        prompt = inputs['input_ids'].shape[-1]
        maximum = body.max_completion_tokens or body.max_tokens
        if prompt + maximum > CTX_SIZE:
            raise ValueError(f'输入 {prompt} + 输出 {maximum} 超过 CTX_SIZE={CTX_SIZE}')
    except Exception as exc:
        lock.release()
        raise HTTPException(400, str(exc)) from exc
    cancelled = threading.Event()
    started = time.monotonic()
    parameters = dict(inputs, max_new_tokens=maximum, do_sample=body.temperature > 0,
                      stopping_criteria=[Cancelled(cancelled)])
    if body.temperature > 0:
        parameters.update(temperature=body.temperature, top_p=body.top_p, top_k=20)
    identifier, created = 'chatcmpl-' + uuid.uuid4().hex, int(time.time())

    def record(count):
        REQUESTS.inc()
        TOKENS.inc(count)
        LATENCY.observe(time.monotonic() - started)

    if not body.stream:
        try:
            with torch.inference_mode():
                output = model.generate(**parameters)
            generated = output[0, prompt:]
            text = processor.decode(generated, skip_special_tokens=True)
            count = len(generated)
            record(count)
            return {'id': identifier, 'object': 'chat.completion', 'created': created, 'model': MODEL_ID,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                                 'finish_reason': 'length' if count == maximum else 'stop'}],
                    'usage': {'prompt_tokens': prompt, 'completion_tokens': count, 'total_tokens': prompt + count}}
        finally:
            lock.release()

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1)
    errors, counts = [], []
    done = threading.Event()

    def worker():
        try:
            with torch.inference_mode():
                output = model.generate(**parameters, streamer=streamer)
            counts.append(output.shape[-1] - prompt)
            record(counts[0])
        except Exception as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc))
        finally:
            done.set()
            lock.release()

    def chunk(delta, finish=None):
        return 'data: ' + json.dumps({'id': identifier, 'object': 'chat.completion.chunk',
            'created': created, 'model': MODEL_ID,
            'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}, ensure_ascii=False) + '\n\n'

    def events():
        try:
            yield chunk({'role': 'assistant'})
            while True:
                try:
                    value = next(streamer)
                    yield chunk({'content': value})
                except Empty:
                    if done.is_set():
                        break
                except StopIteration:
                    # End marker precedes worker cleanup. Wait for counts or exception.
                    done.wait()
                    break
            if errors:
                yield 'data: ' + json.dumps({'error': {'message': errors[0]}}) + '\n\n'
            else:
                yield chunk({}, 'length' if counts and counts[0] == maximum else 'stop')
            yield 'data: [DONE]\n\n'
        finally:
            cancelled.set()

    thread = threading.Thread(target=worker, daemon=True)
    try:
        thread.start()
    except Exception:
        lock.release()
        raise
    return StreamingResponse(events(), media_type='text/event-stream')


if __name__ == '__main__':
    # Single worker: one model copy, one GPU request at a time.
    uvicorn.run(app, host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '8081')))
