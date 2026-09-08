#!/usr/bin/env python3
"""Tool-free JSON inference over an explicitly configured chat-completions API.

This process has no agent harness, MCP tools, plugins, or filesystem operations.
Input is read from stdin; only the JSON content is returned on stdout.
"""
import argparse
import json
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from config import get_config


def infer(prompt, model, timeout=80):
    endpoint = get_config('AGY_MEMORY_INFERENCE_URL')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(infer(sys.stdin.read(), args.model), ensure_ascii=False))
    except Exception as error:
        # Avoid logging provider responses or authorization headers.
        print(f'Memory inference failed ({type(error).__name__}); check endpoint/model configuration.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
