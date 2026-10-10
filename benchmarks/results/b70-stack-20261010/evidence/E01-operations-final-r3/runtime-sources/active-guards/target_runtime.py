"""Reviewed vLLM 0.31 port. Deliberately do not apply legacy runner/cache patches."""
import hashlib
import inspect
import textwrap
from pathlib import Path
import vllm
from vllm.logger import init_logger
from .gdn_rows import HostRows
from .gdn_target_patch import transformed_build

logger = init_logger('vllm.exl3xpu.port')
SOURCE_HASHES = {'v1/attention/backends/gdn_attn.py': '496f383285de2fe92626091309a56a96a64b496f37d5585f8b7b55d0260e1a84',
 'v1/worker/gpu/model_states/mamba_hybrid.py': '1ddd7dc921d74789e77d3319dc167820ad7fc47ed51d8d9a35ba3938f5c6a079',
 'platforms/xpu.py': 'd13a446c60c0cc3d16902f75170646d87d6e4b5f153725290a2a4830b0b6b5b2',
 'v1/core/single_type_kv_cache_manager.py': '3a3cb20bd6aa1e0ad46471fc9e5e01ee5797ab4bb759e00ef5fb309d2830db13',
 'v1/worker/xpu_model_runner.py': '07fa0390d28f2b8c3c3c0019eb9c3b126e2397aea268ba47c889f06b423deb23',
 'v1/worker/xpu_worker.py': '3acd18a33f53b8da5a322079afc61ae7b1247212a7c881ab9aa4101a287c2ca5',
 'model_executor/models/qwen3_5_mtp.py': '04be8e53afff3c9e27198d18ad66bdee787b58c8c294326de1e907f9612a8241',
 'model_executor/models/b70_mtp_policy.py': 'c7c75d6b818c1cb0f1846f752060650af399f832e0dea3e8297ec274f2cfee3f',
 '_xpu_ops.py': '713f181dcfc72814e227f9e1ffe8878b03cbfe524b0be60b3fc261bdb76cd448'}


def install():
    from vllm.v1.attention.backends import gdn_attn
    cls = gdn_attn.GDNAttentionMetadataBuilder
    if getattr(cls, '_exl3_target_patched', False):
        return
    root = Path(vllm.__file__).parent
    actual = {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise RuntimeError(f'Unknown target runtime for EXL3 port: {actual}')
    source, sites = transformed_build(textwrap.dedent(inspect.getsource(cls.build)))
    ns = dict(vars(gdn_attn), _Exl3HostRows=HostRows)
    exec(compile(source, f'<EXL3 target GDN {gdn_attn.__file__}>', 'exec'), ns)
    cls.build = ns['build']
    cls._exl3_target_patched = True
    logger.info('EXL3 target GDN semantic row selection: %s; V2 Mamba metadata uses this builder; '
                'native block64/sparse retirement retained; no V1 align or old attention override', sites)
    from . import attention_dispatch
    attention_dispatch.install()
