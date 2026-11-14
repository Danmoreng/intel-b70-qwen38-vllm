"""Reviewed vLLM 0.33 port. Deliberately do not apply legacy runner/cache patches."""
import hashlib
import inspect
import textwrap
from pathlib import Path
import vllm
from vllm.logger import init_logger
from .gdn_rows import HostRows
from .gdn_target_patch import transformed_build

logger = init_logger('vllm.exl3xpu.port')
SOURCE_HASHES = {'v1/attention/backends/gdn_attn.py': '90f67d9fa8b06fd531579ed1198678c3251baabb47a7b0c9dfdc828d4361ec5e', 'v1/worker/gpu/model_states/mamba_hybrid.py': '262de489e3c02a2b1b93acd2a0d88ddced82ecebe050ca95ea91ba3fe501e6f4', 'platforms/xpu.py': 'cffd8bb94d6d4d6997d30299b91e3d2f249372cad43cb8fa06937e8a111bad18', 'v1/core/single_type_kv_cache_manager.py': '4b8c34ff759855b94a5a3a4a3a1b4784077f27077539e01509c1b3f72e75455f', 'v1/worker/xpu_model_runner.py': '3b079556316176408068357974a9c7b020c827a50781d264123314ce30f42cc6', 'v1/worker/xpu_worker.py': '7846750f1f5a563257ec785e75dac508894ddd8e5f200a466d0940986ed83213', 'model_executor/models/qwen3_5_mtp.py': '39068e51c14f5c1dbfd5e9683a81e5c36c1726709c3037928cc7914c3e9ef303', 'model_executor/models/b70_mtp_policy.py': 'c7c75d6b818c1cb0f1846f752060650af399f832e0dea3e8297ec274f2cfee3f', '_xpu_ops.py': 'a42c3b1f71197e85a29ce21a75711f32ed8d47892ac59431050e5a7442c14410', 'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py': 'c6c037ad94d217aa2710cf29d25d8fa6e3856d1be9fc67fa8bd8681b1f5417f9'}


def install():
    if vllm.__version__ != '0.33.0':
        raise RuntimeError('Unsupported EXL3 runtime version')
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
