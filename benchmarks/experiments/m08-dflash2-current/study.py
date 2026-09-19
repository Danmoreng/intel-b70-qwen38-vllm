#!/usr/bin/env python3
"""Candidate-only DFlash2 qualification and frozen coding rerun.

Production is borrowed only while holding the shared experiment lock. The
systemd launcher also calls recover on failure, cancellation and normal exit.
"""
import argparse
import fcntl
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
NAME = 'b70-m08-dflash2-arm'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diag = load('diag', REPO/'scripts/run-diagnostics.py')
prepare = load('prepare', ROOT/'prepare.py')


def save(path, value):
    diag.save(path, value)


def verify(run):
    manifest = json.loads((run/'manifest.json').read_text())
    for directory, key in [('frozen', 'source_files'), ('harness', 'harness_files'),
                           ('dependencies', 'dependency_files')]:
        if prepare.hashes(run/directory) != manifest[key]:
            raise RuntimeError(f'Frozen {directory} changed')
    draft = Path(manifest['draft_path'])
    if prepare.digest(draft/'model.safetensors') != manifest['draft_sha256']:
        raise RuntimeError('Draft weights changed')
    if prepare.digest(draft/'config.json') != manifest['draft_config_sha256']:
        raise RuntimeError('Draft config changed')
    production = diag.production_inspect()
    if production['Image'] != manifest['control']:
        raise RuntimeError('Production image no longer matches historical reference')
    reference = json.loads((Path(manifest['reference_run'])/'production-inspect.json').read_text())
    if production['Config']['Cmd'] != reference['Config']['Cmd']:
        raise RuntimeError('Production arguments changed since reference')
    relevant_env = lambda value: sorted(e for e in value['Config']['Env'] if e.startswith(
        ('B70_', 'VLLM_', 'ZE_', 'PYTORCH_', 'HF_HUB_OFFLINE=', 'PYTHONPATH=')))
    if relevant_env(production) != relevant_env(reference):
        raise RuntimeError('Production inference environment changed since reference')
    if int(diag.POWER_CAP.read_text()) != 180000000:
        raise RuntimeError('Power cap changed')
    return manifest, production


def command(production, manifest, out):
    args = production['Config']['Cmd'].copy()
    args[args.index('--dtype')+1] = manifest['candidate_dtype']
    args[args.index('--speculative-config')+1] = json.dumps(manifest['candidate_spec'])
    cmd = diag.engine_command(production, name=NAME, image=manifest['candidate'],
                              evidence=out, arguments=args)
    at = cmd.index(manifest['candidate'])
    cmd[at:at] = ['-v', manifest['draft_path']+':/draft:ro',
                  '-e', 'VLLM_SERVER_DEV_MODE=1', '-e', 'VLLM_USE_V2_MODEL_RUNNER=1']
    return cmd


def recover(run):
    marker = run/'production-owned'
    if not marker.exists():
        return
    for file in run.rglob('sandbox-scope.json'):
        scope = json.loads(file.read_text())['runId']
        if not scope or any(c not in '0123456789abcdef' for c in scope):
            raise RuntimeError('Invalid sandbox ID')
        subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGTERM',
                        f'local-ai-b70-job-{scope}.scope'], capture_output=True, timeout=20)
    diag.stop_container(NAME, None)
    subprocess.run(['python3', str(REPO/'scripts/restore-production.py')], check=True,
                   timeout=850, env={**os.environ, 'B70_RUN_DIR': str(run)})
    marker.unlink()


def reset(out):
    result = diag.api('/reset_prefix_cache', {})
    save(out/'cache-reset.json', result)
    if result != {'success': True}:
        raise RuntimeError('Cache reset failed')


def screen(run, out, manifest):
    # An actual load at the unchanged ceiling is the first capacity gate.
    model = diag.api('/v1/models')['data'][0]
    if model['max_model_len'] != manifest['max_context']:
        raise RuntimeError('Capacity does not match reference')
    before, _ = diag.prometheus()
    result = diag.api('/v1/chat/completions', {
        'model': 'Qwen3.8-27B', 'messages': [{'role': 'user', 'content':
        'Write a Python function for a bounded LRU cache and explain how you would test it.'}],
        'temperature': 0, 'max_tokens': 256,
        'chat_template_kwargs': {'enable_thinking': False}}, timeout=180)
    save(out/'smoke.json', result)
    time.sleep(1)
    after, _ = diag.prometheus()
    accepted = diag.metric_total(after, 'vllm:spec_decode_num_accepted_tokens_total')-diag.metric_total(before, 'vllm:spec_decode_num_accepted_tokens_total')
    drafted = diag.metric_total(after, 'vllm:spec_decode_num_draft_tokens_total')-diag.metric_total(before, 'vllm:spec_decode_num_draft_tokens_total')
    if accepted <= 0 or drafted <= 0:
        raise RuntimeError(f'DFlash acceptance gate failed: {accepted}/{drafted}')
    save(out/'acceptance.json', {'accepted': accepted, 'drafted': drafted})
    with (out/'prefix-state.log').open('w') as log:
        subprocess.run(['python3', str(REPO/'scripts/check-prefix-state.py'),
                        '--root', diag.EXPERIMENT_URL, '--output', str(out/'prefix-state.json')],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    # Reuse the actual largest prompt from the reference trajectory. Only its
    # output cap is reduced for this capacity check; it is not a speed result.
    reference = json.loads((run/'reference-task-result.json').read_text())
    largest = max(reference['requests'], key=lambda row: row.get('prompt_tokens', 0))
    body = json.loads((Path(manifest['reference_run'])/manifest['reference_arm']/f"request-{largest['index']}.json").read_text())
    body.update(stream=False, max_tokens=32)
    body.pop('stream_options', None)
    result = diag.api('/v1/chat/completions', body, timeout=1200)
    save(out/'long-context.json', result)
    if result.get('usage', {}).get('prompt_tokens') != largest['prompt_tokens']:
        raise RuntimeError('Longest reference prompt token count changed')
    if not result.get('choices'):
        raise RuntimeError('Longest reference prompt produced no completion')
    smoke = out/'agent-smoke'
    smoke.mkdir()
    shutil.copytree(run/'harness', smoke/'harness')
    shutil.copytree(run/'frozen', smoke/'workspace', symlinks=True)
    (smoke/'dependencies').symlink_to(run/'dependencies', target_is_directory=True)
    smoke_manifest = {**manifest, 'limits': {'wall_seconds': 180, 'model_requests': 8,
                                            'completion_tokens_total': 2048}}
    save(smoke/'manifest.json', smoke_manifest)
    (smoke/'harness/task.md').write_text('Use the read tool to inspect /workspace/dashboard/package.json. Report the package name. Do not edit files or run tests.\n')
    (smoke/'harness/followup.md').write_text('Use the read tool to inspect /workspace/dashboard/packages/server/package.json. Report the package name. Do not edit files or run tests.\n')
    with (smoke/'driver.log').open('w') as log:
        subprocess.run(['node', str(smoke/'harness/run-task.mjs'), str(smoke), str(smoke), 'dflash2-smoke'],
                       stdout=log, stderr=subprocess.STDOUT, timeout=240, check=True)
    task = json.loads((smoke/'task-result.json').read_text())
    events = (smoke/'pi-events.jsonl').read_text()
    if not task['initial_and_followup_complete'] or 'tool_execution_end' not in events:
        raise RuntimeError('Isolated tool/continuation smoke did not complete')
    if not task['requests'] or any(row['status'] != 'ok' for row in task['requests']):
        raise RuntimeError('Native metric accounting did not pass')
    save(run/'preflight.json', {'ready': True, 'startup': True,
        'nonzero_draft_acceptance': True, 'prefix_state': True,
        'longest_reference_prompt': largest['prompt_tokens'], 'isolated_agent': True,
        'manifest_sha256': prepare.digest(run/'manifest.json'), 'evidence': str(out)})


def coding(run, out, manifest):
    shutil.copytree(run/'frozen', out/'workspace', symlinks=True)
    for cmd in [['git', 'init', '-q'], ['git', 'add', 'dashboard'],
                ['git', '-c', 'user.name=Benchmark', '-c', 'user.email=benchmark@localhost',
                 'commit', '-qm', 'Frozen coding baseline']]:
        subprocess.run(cmd, cwd=out/'workspace', check=True, stdout=subprocess.DEVNULL)
    with (out/'warmup.log').open('w') as log:
        subprocess.run(['python3', str(REPO/'scripts/benchmark.py'), '--root', diag.EXPERIMENT_URL,
                        '--contexts', '8192', '--output-tokens', '64', '--repeats', '1',
                        '--nonce', 'excluded-coding-warmup', '--output-dir', str(out/'warmup')],
                       check=True, timeout=900, stdout=log, stderr=subprocess.STDOUT)
    reset(out)
    with diag.TelemetrySampler(out/'telemetry.jsonl'), (out/'driver.log').open('w') as log:
        return subprocess.run(['node', str(run/'harness/run-task.mjs'), str(out), str(run), 'dflash2'],
                              stdout=log, stderr=subprocess.STDOUT,
                              timeout=manifest['limits']['wall_seconds']+90).returncode


def evaluate(run, out, manifest):
    checks = (run/'harness/baseline-checks.json').read_text()
    evaluator = run/'harness/check-workspace.mjs'
    results = {}
    for label, commands in [('validation', checks), ('acceptance', json.dumps([['node', '/runtime/acceptance.mjs']]))]:
        with (out/f'{label}.log').open('w') as log:
            result = subprocess.run(['node', str(evaluator), str(out/'workspace'), str(run/'dependencies'), commands, 'continue'],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1250)
        results[label+'_exit_code'] = result.returncode
    text = (out/'acceptance.log').read_text()
    pos = text.find('{\n  "schema": 1,')
    if pos >= 0:
        results['acceptance'] = json.JSONDecoder().raw_decode(text[pos:])[0]
    clean = re.sub(r'\x1b\[[0-9;]*m', '', (out/'validation.log').read_text())
    warnings = [line.split('[warn] ',1)[1] for line in clean.splitlines()
                if '[warn] ' in line and 'Code style issues' not in line]
    results['format_warnings'] = warnings
    results['only_preexisting_format_failure'] = (
        results['validation_exit_code'] == 1 and clean.count('CHECK_FAILED') == 1
        and len(warnings) > 0 and set(warnings) <= set(manifest['baseline_format_failures']))
    results['checks_pass'] = results['validation_exit_code'] == 0 or results['only_preexisting_format_failure']
    results['task_acceptance_pass'] = results['acceptance_exit_code'] == 0
    save(out/'validation.json', {**results, 'baseline_format_failures': manifest['baseline_format_failures']})
    with (out/'result.diff').open('w') as stream:
        subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', 'diff', 'HEAD', '--', 'dashboard'],
                       cwd=out/'workspace', stdout=stream, check=True)
    old = load('oldstudy', run/'harness/run-study.py')
    state = {'status': 'complete-not-promoted', 'arms': [{'label': 'dflash2',
             'image': manifest['candidate'], 'directory': str(out), 'validation': results}]}
    old.summarize(run, state)
    candidate = json.loads((run/'summary.json').read_text())['arms'][0]
    reference = next(arm for arm in json.loads((run/'reference-summary.json').read_text())['arms']
                     if arm['arm'] == 'control')
    save(run/'comparison.json', {'method': manifest['comparison'],
         'reference': reference, 'candidate': candidate,
         'limitations': ['Historical reference, no interleaved control rerun',
                         'Same tasks, different adaptive trajectories',
                         'Candidate uses BF16 instead of FP16 activations']})
    (run/'RESULTS.md').write_text(
        '# DFlash2 versus recorded production coding run\n\n'+manifest['comparison']+'\n\n'
        'See comparison.json for task outcomes, weighted context bands, native phase times, '
        'prefix reuse, speculative acceptance, GPU energy and independent checks. '
        'No production promotion is implied.\n')


def require_gate(run):
    gate = json.loads((run/'preflight.json').read_text())
    if not gate.get('ready') or gate.get('manifest_sha256') != prepare.digest(run/'manifest.json'):
        raise RuntimeError('DFlash serving gates have not passed for this manifest; long run blocked')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'coding', 'recover', 'verify'])
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.mode == 'recover':
        recover(run)
        return
    manifest, production = verify(run)
    if args.mode == 'verify':
        print('Frozen source, harness, dependencies, model and reference profile verified.')
        return
    if args.mode == 'coding':
        require_gate(run)
    lock = Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    diag.ensure_idle(diag.PRODUCTION_URL)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    out = run/(('probe-' if args.mode == 'preflight' else 'arm-')+time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir()
    state = {'status': 'loading', 'mode': args.mode, 'evidence': str(out)}
    save(run/'state.json', state)
    save(run/'production-inspect.json', production)
    cmd = command(production, manifest, out)
    save(out/'engine-command.json', cmd)
    (run/'production-owned').write_text('M08 exclusive experiment\n')
    engine = None
    try:
        subprocess.run(['systemctl', '--user', 'stop', diag.PRODUCTION_SERVICE], check=True, timeout=120)
        with (out/'engine.log').open('w') as log:
            engine = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            diag.wait_ready(engine)
            state['status'] = args.mode
            save(run/'state.json', state)
            if args.mode == 'preflight':
                screen(run, out, manifest)
                state['status'] = 'ready-for-coding'
            else:
                state['agent_exit_code'] = coding(run, out, manifest)
                state['status'] = 'coding-finished-awaiting-evaluation'
    except BaseException as error:
        state['status'] = 'blocked' if args.mode == 'preflight' else 'failed'
        state['error'] = repr(error)
        if args.mode == 'preflight':
            save(run/'preflight.json', {'ready': False, 'error': repr(error), 'evidence': str(out)})
        raise
    finally:
        save(run/'state.json', state)
        diag.stop_container(NAME, engine)
        recover(run)
    if args.mode == 'coding':
        evaluate(run, out, manifest)
        state['status'] = 'complete-not-promoted'
        save(run/'state.json', state)


if __name__ == '__main__':
    main()
