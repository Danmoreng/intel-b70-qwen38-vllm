#!/usr/bin/env python3
"""Recover a live but stalled production service; never start an inactive service."""
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from zoneinfo import ZoneInfo

SERVICE = 'b70-qwen38-vllm.service'
BASE_URL = 'http://127.0.0.1:8081'
STATE = Path.home() / '.local/state/b70-qwen38-vllm/inference-watchdog.json'
STARTUP_GRACE = 15 * 60
STALL_SECONDS = 10 * 60
GPU_STALL_SECONDS = 60
COOLDOWN = 15 * 60


def parse_metrics(text):
    values = {}
    wanted = {'num_requests_running', 'num_requests_waiting', 'generation_tokens_total',
              'prompt_tokens_total', 'request_success_total'}
    for line in text.splitlines():
        match = re.match(r'^vllm:(\w+)(?:\{[^}]*\})?\s+(\S+)', line)
        if match and match[1] in wanted:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid progress metric')
            values[match[1]] = values.get(match[1], 0) + value
    if not {'num_requests_running', 'generation_tokens_total', 'prompt_tokens_total'} <= values.keys():
        raise ValueError('Required progress metrics missing')
    return values


def observe(previous, invocation, age, now, metrics, healthy, gpu_fault, allowed=True):
    """Return new persistent state and a bounded recovery decision (no side effects)."""
    restarts = [t for t in previous.get('restarts', []) if 0 <= now - t < 3600]
    if previous.get('invocation') != invocation or now < previous.get('seen_at', now):
        previous = {}
    state = {**previous, 'invocation': invocation, 'seen_at': now, 'restarts': restarts}
    if age < STARTUP_GRACE and metrics is None and not healthy:
        state.update(progress_at=now, unhealthy_at=now, gpu_fault=False)
        return state, None
    if gpu_fault:
        state['gpu_fault'] = True
    if metrics is not None:
        signature = [metrics.get(name, 0) for name in
                     ('generation_tokens_total', 'prompt_tokens_total', 'request_success_total')]
        busy = metrics['num_requests_running'] + metrics.get('num_requests_waiting', 0) > 0
        if signature != state.get('signature') or not busy:
            state.update(progress_at=now)
            # A new fault remains pending until subsequent real progress.
            if not gpu_fault:
                state['gpu_fault'] = False
        state['signature'] = signature
        state['busy'] = busy
    else:
        state.setdefault('progress_at', now)
    if healthy:
        state['unhealthy_at'] = now
    else:
        state.setdefault('unhealthy_at', now)
    reason = None
    stalled = now - state.get('progress_at', now)
    if state.get('busy') and state.get('gpu_fault') and stalled >= GPU_STALL_SECONDS:
        reason = 'vLLM GPU job fault and no inference progress for 60 seconds'
    elif state.get('busy') and stalled >= STALL_SECONDS:
        reason = 'Active inference has made no progress for 10 minutes'
    elif not healthy and now - state['unhealthy_at'] >= STALL_SECONDS:
        reason = 'Model API has been unavailable for 10 minutes after startup grace'
    if not allowed or len(restarts) >= 3 or (restarts and now - restarts[-1] < COOLDOWN):
        reason = None
    if reason:
        state['restarts'] = restarts + [now]
        state['last_reason'] = reason
    return state, reason


def fetch(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=3) as response:
        return response.read(2 * 1024 * 1024).decode()


def save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = STATE.with_suffix('.new')
    with temporary.open('w') as out:
        os.chmod(temporary, 0o600)
        json.dump(state, out, indent=2)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(STATE)


def capture_failure(state):
    directory = STATE.parent / ('recovery-' + str(int(state['seen_at'])))
    directory.mkdir(mode=0o700, exist_ok=True)
    (directory / 'state.json').write_text(json.dumps(state, indent=2))
    for name, command in (
        ('kernel.log', ['journalctl', '-k', '-n', '200', '--no-pager']),
        ('engine.log', ['journalctl', '--user', '-u', SERVICE, '-n', '200', '--no-pager']),
    ):
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=5)
            (directory / name).write_text(output)
        except (OSError, subprocess.SubprocessError) as error:
            (directory / name).write_text(type(error).__name__)
    # Keep bounded diagnostics, never prompt text or an unbounded GPU dump.
    for old in sorted(STATE.parent.glob('recovery-*'), key=lambda p: p.name)[0:-8]:
        if old.is_dir() and all(child.is_file() for child in old.iterdir()):
            for child in old.iterdir():
                child.unlink()
            old.rmdir()


def main():
    info = subprocess.check_output(['systemctl', '--user', 'show', SERVICE,
        '-p', 'ActiveState', '-p', 'InvocationID', '-p', 'ExecMainStartTimestampMonotonic'], text=True, timeout=5)
    props = dict(line.split('=', 1) for line in info.splitlines())
    if props['ActiveState'] != 'active':
        return  # An operator stop must remain stopped; systemd owns actual crashes.
    now = time.time()
    age = time.monotonic() - int(props['ExecMainStartTimestampMonotonic']) / 1_000_000
    try:
        previous = json.loads(STATE.read_text())
    except (FileNotFoundError, ValueError):
        previous = {}
    try:
        metrics = parse_metrics(fetch('/metrics'))
    except (OSError, ValueError):
        metrics = None
    try:
        fetch('/health')
        healthy = True
    except OSError:
        healthy = False
    try:
        kernel = subprocess.check_output(['journalctl', '-k', '--since', '@' + str(int(max(now - 180, now - age))), '-o', 'cat',
            '--no-pager'], text=True, stderr=subprocess.DEVNULL, timeout=5)
        fault = any('Timedout job:' in line and 'VLLM::EngineCor' in line for line in kernel.splitlines())
    except (OSError, subprocess.SubprocessError):
        fault = False  # The generic progress check also works without kernel-log access.
    berlin = datetime.datetime.now(ZoneInfo('Europe/Berlin'))
    allowed = berlin.hour * 60 + berlin.minute < 22 * 60 + 45
    state, reason = observe(previous, props['InvocationID'], age, now, metrics, healthy, fault, allowed)
    save(state)
    if reason:
        capture_failure(state)
        print(json.dumps({'action': 'restart', 'reason': reason, 'metrics': metrics}), flush=True)
        # try-restart preserves an operator stop racing with this observation.
        current = subprocess.check_output(['systemctl', '--user', 'show', SERVICE,
            '-p', 'InvocationID', '--value'], text=True, timeout=5).strip()
        if current == props['InvocationID']:
            subprocess.run(['systemctl', '--user', 'try-restart', SERVICE], check=True, timeout=90)


if __name__ == '__main__':
    main()
