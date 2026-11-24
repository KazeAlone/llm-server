"""Dependency-free chat/monitor UI and same-origin API proxy (no key in argv)."""
import argparse
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse

ALLOWED = {'/v1/models', '/v1/chat/completions', '/health', '/metrics'}


def serve(host, port, upstream):
    target = urlparse(upstream)
    if target.scheme != 'http' or not target.hostname:
        raise ValueError('仅支持局域网/本机 http 后端地址')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.route()

        def do_POST(self):
            self.route()

        def route(self):
            if self.path == '/' and self.command == 'GET':
                data = Path(__file__).with_name('console.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            expected = os.environ.get('API_KEY', '')
            if expected and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + expected):
                self.send_error(401)
                return
            if self.path == '/__gpu' and self.command == 'GET':
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw',
                                             '--format=csv'], capture_output=True, text=True, timeout=10)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.send_error(503, str(exc))
                    return
                data = result.stdout.encode() or result.stderr.encode()
                self.send_response(200 if result.returncode == 0 else 503)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path not in ALLOWED or (self.command == 'POST' and self.path != '/v1/chat/completions'):
                self.send_error(404)
                return
            length = int(self.headers.get('Content-Length', '0'))
            if length > 32 * 1024 * 1024:
                self.send_error(413)
                return
            connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=3600)
            headers = {'Content-Type': 'application/json', 'Authorization': self.headers.get('Authorization', '')}
            sent = False
            try:
                connection.request(self.command, self.path, self.rfile.read(length), headers)
                response = connection.getresponse()
                sent = True
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                if not sent:
                    self.send_error(502, "Backend unavailable or loading")
            finally:
                connection.close()

    print(f'UI http://{host}:{port} → {upstream}', flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8180)
    parser.add_argument('--upstream', required=True)
    args = parser.parse_args()
    serve(args.host, args.port, args.upstream)
