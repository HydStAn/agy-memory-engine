#!/usr/bin/env python3
"""Dual-mode JSON inference for memory extraction and consolidation.

Supports both:
1. Tool-free HTTP chat-completions API when AGY_MEMORY_INFERENCE_URL is configured.
2. Graceful fallback to native Antigravity CLI (agy --print) when unconfigured.

Input is read from stdin; only valid JSON content is returned on stdout.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from config import AGY_BIN, get_config


def _infer_http(prompt, model, endpoint, timeout=80):
    parsed = urlparse(endpoint)
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1')):
        raise ValueError('Set AGY_MEMORY_INFERENCE_URL to a trusted HTTPS chat-completions endpoint or a loopback HTTP endpoint')
    selected = get_config('AGY_MEMORY_INFERENCE_MODEL', model)
    headers = {'Content-Type': 'application/json'}
    token = get_config('AGY_MEMORY_INFERENCE_KEY')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    body = {'model': selected, 'messages': [{'role': 'user', 'content': prompt}],
            'response_format': {'type': 'json_object'}, 'stream': False}
    request = Request(endpoint, data=json.dumps(body).encode(), headers=headers, method='POST')
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Inference response exceeds 2 MB')
    message = json.loads(raw)['choices'][0]['message']
    if message.get('tool_calls') or message.get('function_call'):
        raise ValueError('Tool calls are not permitted in memory inference')
    result = json.loads(message['content'])
    if not isinstance(result, dict):
        raise ValueError('Inference must return a JSON object')
    return result


def _infer_cli(prompt, model, timeout=80):
    env = dict(os.environ, AGY_INTERNAL_INVOCATION='1', AGY_SAGE_DISABLED='1')
    cmd = [
        AGY_BIN,
        '--print',
        prompt,
        '--model',
        model,
        '--dangerously-skip-permissions',
        '--disable-slash-commands',
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError('CLI inference timed out') from error
    except OSError as error:
        raise RuntimeError(f'Cannot launch CLI inference ({AGY_BIN})') from error

    if res.returncode != 0 and '--disable-slash-commands' in (res.stderr or ''):
        cmd = [
            AGY_BIN,
            '--print',
            prompt,
            '--model',
            model,
            '--dangerously-skip-permissions',
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError('CLI inference timed out') from error
        except OSError as error:
            raise RuntimeError(f'Cannot launch CLI inference ({AGY_BIN})') from error

    if res.returncode != 0:
        raise RuntimeError(f'Antigravity CLI failed with code {res.returncode}')

    out = res.stdout.strip()
    if not out:
        raise ValueError('Antigravity CLI returned empty output')

    match = re.search(r'\{.*\}', out, re.DOTALL)
    if not match:
        raise ValueError('Antigravity CLI output contains no JSON object')

    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError('Antigravity CLI output is not valid JSON') from error

    if not isinstance(result, dict):
        raise ValueError('Inference must return a JSON object')
    return result


def infer(prompt, model, timeout=80):
    endpoint = (get_config('AGY_MEMORY_INFERENCE_URL') or '').strip()
    if endpoint:
        return _infer_http(prompt, model, endpoint, timeout=timeout)
    return _infer_cli(prompt, model, timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(infer(sys.stdin.read(), args.model), ensure_ascii=False))
    except Exception as error:
        # Avoid logging provider responses or authorization headers.
        print(f'Memory inference failed ({type(error).__name__}); check endpoint/model/CLI configuration.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
