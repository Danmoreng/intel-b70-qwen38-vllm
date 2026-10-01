#!/usr/bin/env python3
"""Guard the reviewed 0.30 MTP hooks. Never silently patch an unknown anchor."""
import argparse
import ast
from pathlib import Path

MARKER='B70_MTP_FORMAT_GUARD_V1'

def replace_once(source, old, new):
    if source.count(old)!=1: raise ValueError(f'Expected exactly one MTP anchor: {old!r}')
    return source.replace(old,new,1)

def guarded_source(source):
    if MARKER in source: return source
    source=replace_once(source,'import torch\n',
                        'import torch\nfrom .b70_mtp_policy import resolve_mtp_policy\n# '+MARKER+'\n')
    source=replace_once(source,'        quant_config = vllm_config.quant_config\n',
                        '        quant_config = vllm_config.quant_config\n'
                        '        self._b70_mtp_policy = resolve_mtp_policy(quant_config, model_config.hf_config)\n'
                        '        logger.info("B70 MTP quantization path: %s", self._b70_mtp_policy.description)\n')
    source=replace_once(source,'        self.quant_config = vllm_config.quant_config\n',
                        '        self.quant_config = vllm_config.quant_config\n'
                        '        self._b70_mtp_policy = resolve_mtp_policy(self.quant_config, vllm_config.model_config.hf_config)\n')
    source=replace_once(source,'if quant_config and quant_config.get_name() not in ("modelopt_fp4",):',
                        'if self._b70_mtp_policy.is_gptq:')
    substitutions={'os.environ.get("B70_MTP_BF16_DRAFT") == "1"':'self._b70_mtp_policy.bf16_draft',
                   'os.environ.get("B70_DRAFT_LMHEAD_INT4") == "1"':'self._b70_mtp_policy.lmhead_int4',
                   'os.environ.get("B70_DRAFT_MTP_INT4") == "1"':'self._b70_mtp_policy.mtp_int4'}
    for old,new in substitutions.items(): source=replace_once(source,old,new)
    ast.parse(source)
    return source

def patch(root):
    models=Path(root)/'model_executor/models'; path=models/'qwen3_5_mtp.py'
    new=guarded_source(path.read_text())
    (models/'b70_mtp_policy.py').write_text(Path(__file__).with_name('b70_mtp_policy.py').read_text())
    path.write_text(new)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vllm-root',required=True)
    patch(p.parse_args().vllm_root)
