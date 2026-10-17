"""Validated production runtime overlay on the immutable EXL3 base release."""
import hashlib
import json
import os
from pathlib import Path


def load_overlay(repo, release, environ=None):
    attestation = release.get('ram_kv_runtime')
    if attestation is None:
        return None
    root = Path(repo)
    path = root / 'config/production_ram_kv.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != attestation['sha256']:
        raise RuntimeError('RAM-KV release manifest changed')
    data = json.loads(raw)
    if data['status'] != 'PRODUCTION' or data['base_image_id'] != release['image_id']:
        raise RuntimeError('RAM-KV runtime release identity mismatch')
    expected = {'kv_admission.py', 'xpu_cpu_offload.py'}
    if set(data['runtime_sha256']) != expected:
        raise RuntimeError('Unexpected RAM-KV runtime modules')
    for name, digest in data['runtime_sha256'].items():
        if hashlib.sha256((root / 'runtime' / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('RAM-KV module changed: ' + name)
    env = os.environ if environ is None else environ
    budget = int(env.get('B70_KV_RAM_GIB', str(data['default_cpu_gib'])))
    if not 1 <= budget <= 12:
        raise ValueError('B70_KV_RAM_GIB must be an integer from 1 to 12 GiB')
    return {'sha256': attestation['sha256'], 'cpu_gib': budget,
            'scheduler': 'kv_admission.CompletionReservedScheduler', 'watermark': '0.05',
            'transfer': {'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
                         'kv_connector_extra_config': {
                             'cpu_bytes_to_use': budget * 1024**3,
                             'spec_name': 'BoundedXPUOffloadingSpec',
                             'spec_module_path': 'xpu_cpu_offload',
                             'blocks_per_chunk': 1, 'eviction_policy': 'arc', 'store_threshold': 0}}}


def serving_arguments(argv, overlay):
    args = list(argv)
    if overlay is None:
        return args
    for flag, value in [('--watermark', overlay['watermark']),
                        ('--scheduler-cls', overlay['scheduler']),
                        ('--kv-transfer-config', json.dumps(overlay['transfer']))]:
        if flag in args:
            index = args.index(flag)
            del args[index:index + 2]
        args += [flag, value]
    return args
