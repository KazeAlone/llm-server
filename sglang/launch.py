"""SGLang entrypoint with key outside argv and local image path allowlist."""
import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse


def check_media(value, image=False):
    if isinstance(value, dict):
        for key, item in value.items():
            check_media(item, image or key in ('image_url', 'image_data', 'video_data', 'audio_data'))
    elif isinstance(value, list):
        for item in value:
            check_media(item, image)
    elif isinstance(value, str) and (value.startswith('file://') or (image and value.startswith('/'))):
        if os.environ.get('VISION', '1') == '0':
            raise ValueError('VISION=0')
        allowed = os.environ.get('MEDIA_PATH', '/')
        if not allowed:
            raise ValueError('本地媒体访问已禁用')
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc not in ('', 'localhost'):
            raise ValueError('file URL 不允许远程主机名')
        path = Path(unquote(parsed.path)).resolve()
        if not path.is_relative_to(Path(allowed).resolve()):
            raise ValueError('媒体位于 MEDIA_PATH 之外')


if __name__ == '__main__':
    from fastapi.responses import JSONResponse
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.entrypoints.http_server import app
    from sglang.srt.utils import kill_process_tree

    @app.middleware('http')
    async def local_media(request, call_next):
        if request.method == 'POST' and request.url.path in ('/v1/chat/completions', '/generate'):
            try:
                body = await request.json()
                if os.environ.get('VISION', '1') == '0':
                    def has_image(value):
                        if isinstance(value, dict):
                            return value.get('type') in ('image_url', 'image', 'video_url') or any(
                                k in ('image_data', 'video_data') or has_image(v) for k, v in value.items())
                        return isinstance(value, list) and any(map(has_image, value))
                    if has_image(body):
                        raise ValueError('VISION=0，图片输入已关闭')
                check_media(body)
            except (ValueError, TypeError) as exc:
                return JSONResponse({'error': {'message': str(exc)}}, status_code=400)
        return await call_next(request)

    load_plugins()
    args = prepare_server_args(sys.argv[1:])
    if args.tokenizer_worker_num != 1:
        raise ValueError('本地媒体中间件要求 tokenizer_worker_num=1')
    args.api_key = os.environ.get('LLM_SERVER_API_KEY') or None
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
