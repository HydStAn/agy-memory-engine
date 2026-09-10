#!/usr/bin/env python3
"""Render separate Stop hook and periodic worker configs without installing them."""
import argparse
import json
import plistlib
import shlex
import sys
from pathlib import Path


def render(output_dir, python_path, repository):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(repository).resolve()
    interpreter = str(Path(python_path).resolve())
    hook = {'memory-auto-sync': {'Stop': [{'type':'command',
        'command': shlex.join([interpreter,str(repo/'scripts/auto_sync_hook.py')]), 'timeout':10}]}}
    (output/'hooks.fragment.json').write_text(json.dumps(hook,indent=2)+'\n')
    plist = {'Label':'local.agy.memory-worker', 'ProgramArguments':[interpreter,str(repo/'memory_worker.py'),'--no-notify'],
             'WorkingDirectory':str(repo), 'StartInterval':60, 'RunAtLoad':True,
             'StandardOutPath':str(output/'worker.stdout.log'), 'StandardErrorPath':str(output/'worker.stderr.log')}
    with (output/'local.agy.memory-worker.plist').open('wb') as handle:
        plistlib.dump(plist,handle)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--python',default=sys.executable)
    args=parser.parse_args()
    print(render(args.output_dir,args.python,Path(__file__).resolve().parent.parent))
