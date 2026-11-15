#!/usr/bin/env python3
"""Bounded diagnostic replay of captured API requests; never a coding score."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'agent-panel'))
import coding_tools as K
sys.path.insert(0, str(HERE.parents[2] / 'scripts'))
from exl3_candidate_worker import profile

SERVICE = 'b70-qwen38-vllm.service'
NAME = 'b70-v033-generation-probe'
BASE = 'http://127.0.0.1:8082'


def ready(base):
    end = time.monotonic() + 900
    while time.monotonic() < end:
        try:
            return K.http_json(base + '/v1/models', timeout=3)
        except OSError:
            time.sleep(2)
    raise TimeoutError('Model readiness: ' + base)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('launch', 'requests', 'out'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--image', required=True)
    p.add_argument('--output-cap', type=int, default=1024, choices=(1024, 2048, 16384),
                   help='Common recorded output cap for every diagnostic arm')
    p.add_argument('--tokenize-first', action='store_true',
                   help='Reproduce the live task harness tokenization before each request')
    p.add_argument('--return-token-ids', action='store_true',
                   help='Record raw generated tokens as well as parsed tool calls')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--enforce-eager', action='store_true', help='Explicit diagnostic override, never a release configuration')
    mode.add_argument('--disable-compile', action='store_true', help='Disable Torch compilation while retaining the original graph mode')
    mode.add_argument('--compile-backend', choices=('eager', 'aot_eager'), help='Diagnostic Dynamo backend with the original compilation and graph modes')
    mode.add_argument('--custom-op', action='append', default=[], choices=('gemma_rms_norm', 'all'), help='Keep Inductor and graphs but select native custom operators explicitly')
    a = p.parse_args()
    assert dt.datetime.now(ZoneInfo('Europe/Berlin')).hour < 21
    source = json.loads(a.launch.read_text())
    actual_profile = profile(a.image)
    assert actual_profile == source['profile'], 'Launch receipt must match the selected image'
    cmd = source['command'][:]
    assert a.image in cmd
    assert cmd[cmd.index('--name') + 1] == 'b70-v033-candidate'
    cmd[cmd.index('--name') + 1] = NAME
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'worker').mkdir()
    destinations = [i + 1 for i, value in enumerate(cmd[:-1]) if value == '-v' and cmd[i + 1].endswith(':/results')]
    assert len(destinations) == 1
    cmd[destinations[0]] = str(root / 'worker') + ':/results'
    if a.enforce_eager:
        cmd.append('--enforce-eager')
    if a.disable_compile or a.compile_backend or a.custom_op:
        index = cmd.index('--compilation-config') + 1
        config = json.loads(cmd[index])
        assert config['cudagraph_mode'] == 'FULL_DECODE_ONLY'
        if a.disable_compile:
            config['mode'] = 0
        elif a.compile_backend:
            config['backend'] = a.compile_backend
        else:
            config['custom_ops'] = ['all'] if a.custom_op == ['all'] else ['none', *('+' + op for op in a.custom_op)]
        cmd[index] = json.dumps(config)
    state = dict(status='RUNNING_DIAGNOSTIC', image_id=a.image, launch=cmd,
                 source_launch_sha256=K.sha(a.launch), enforce_eager=a.enforce_eager,
                 disable_compile=a.disable_compile, runner_sha256=K.sha(Path(__file__)),
                 compile_backend=a.compile_backend,
                 custom_ops=a.custom_op,
                 output_cap=a.output_cap, tokenize_first=a.tokenize_first,
                 return_token_ids=a.return_token_ids, requests=[],
                 scope='Captured messages/tools/seeds/sampling are fixed across arms, with the common output cap and every override recorded separately. No live tools, repaired solution, replacement task score or release approval.')

    def save():
        (root / 'result.json').write_text(json.dumps(state, indent=2) + '\n')

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda n, f: (_ for _ in ()).throw(KeyboardInterrupt(str(n))))
    with Path('/tmp/b70-v033-qualification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = urllib.request.urlopen('http://127.0.0.1:8081/metrics', timeout=5).read().decode()
        for line in raw.splitlines():
            if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{')):
                assert float(line.split()[-1]) == 0, line
        production = json.loads(subprocess.check_output(['docker', 'inspect', 'b70-qwen38-vllm'], text=True))[0]['Image']
        state['production_image_id'] = production
        assert subprocess.run(['docker', 'inspect', NAME], capture_output=True).returncode != 0
        state['started_utc'] = dt.datetime.now(dt.UTC).isoformat()
        save()
        subprocess.run(['systemctl', '--user', 'stop', SERVICE], check=True, timeout=120)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, timeout=60)
            ready(BASE)
            smoke = dict(model='Qwen3.8-27B', messages=[dict(role='user', content='Count from 1 to 100, separated by commas.')],
                         temperature=0, max_tokens=64, chat_template_kwargs={'enable_thinking': False})
            K.http_json(BASE + '/v1/chat/completions', smoke, 180)
            for index in range(3):
                original = a.requests / f'request-{index:03d}.json'
                request = json.loads(original.read_text())
                request['max_tokens'] = a.output_cap
                if a.return_token_ids:
                    request['return_token_ids'] = True
                (root / f'request-{index:03d}.json').write_text(json.dumps(request, indent=2) + '\n')
                if a.tokenize_first:
                    rendered_messages = []
                    for original_message in request['messages']:
                        message = dict(original_message)
                        reasoning = message.pop('reasoning_content', None)
                        if reasoning is not None and message.get('reasoning') is None:
                            message['reasoning'] = reasoning
                        rendered_messages.append(message)
                    rendered = K.http_json(BASE + '/tokenize', dict(
                        model=request['model'], messages=rendered_messages, tools=request['tools'],
                        chat_template_kwargs=request['chat_template_kwargs'], add_generation_prompt=True), 120)
                    (root / f'tokenize-{index:03d}.json').write_text(json.dumps(rendered, indent=2) + '\n')
                old = K.snapshot(BASE)
                start = time.monotonic()
                response = K.http_json(BASE + '/v1/chat/completions', request,
                                       min(900, max(180, a.output_cap // 15 + 60)))
                new = K.snapshot(BASE)
                (root / f'response-{index:03d}.json').write_text(json.dumps(response, indent=2) + '\n')
                if a.return_token_ids:
                    token_ids = response['choices'][0]['token_ids']
                    assert token_ids and len(token_ids) == response['usage']['completion_tokens']
                    decode = K.http_json(BASE + '/detokenize', dict(model=request['model'], tokens=token_ids), 120)
                    (root / f'raw-output-{index:03d}.json').write_text(json.dumps(decode, indent=2) + '\n')
                message = response['choices'][0]['message']
                text = (message.get('reasoning') or message.get('reasoning_content') or '') + (message.get('content') or '')
                count = 0
                while text.endswith('duct' * (count + 1)):
                    count += 1
                state['requests'].append(dict(index=index, source_request_sha256=K.sha(original),
                    seconds=time.monotonic() - start, usage=response['usage'],
                    finish_reason=response['choices'][0]['finish_reason'], trailing_duct_repetitions=count,
                    response_sha256=hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest(),
                    native_delta={key: new[key] - old[key] for key in old}))
                save()
                print('OBSERVED', index, response['choices'][0]['finish_reason'], 'duct suffix', count, flush=True)
            state['status'] = 'COMPLETED_FROZEN_REQUEST_DIAGNOSTIC'
        except BaseException as exc:
            state.update(status='FAILED_DIAGNOSTIC', error=repr(exc))
            raise
        finally:
            subprocess.run(['docker', 'stop', '--time', '90', NAME], capture_output=True, timeout=105)
            logs = subprocess.run(['docker', 'logs', NAME], capture_output=True, text=True)
            (root / 'worker.log').write_text(logs.stdout + logs.stderr)
            subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True, timeout=60)
            subprocess.run(['systemctl', '--user', 'start', SERVICE], check=True, timeout=120)
            ready('http://127.0.0.1:8081')
            actual = json.loads(subprocess.check_output(['docker', 'inspect', 'b70-qwen38-vllm'], text=True))[0]['Image']
            assert actual == production
            state['production_restored'] = True
            state['finished_utc'] = dt.datetime.now(dt.UTC).isoformat()
            save()
            print('RESTORED_PRODUCTION', flush=True)


if __name__ == '__main__':
    main()
